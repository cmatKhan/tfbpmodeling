import json
import logging
import os
import pickle
import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from patsy import PatsyError, dmatrix
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

from tfbpmodeling.stratification_classification import (
    stratification_classification,
)
from tfbpmodeling.stratified_cv_r2 import stratified_cv_r2

logger = logging.getLogger("main")


class ModelingInputData:
    """
    Container for response and predictor data used in modeling transcription factor
    perturbation experiments.

    This class handles:
        - Validation and synchronization of response and predictor DataFrames
        based on a shared feature identifier.
        - Optional blacklisting of features, including the perturbed
            transcription factor.
        - Optional feature selection based on the top N strongest binding signals
            (as ranked from a specific TF column in the predictor matrix).
        - Application of masking logic to restrict modeling to selected features.

    """

    def __init__(
        self,
        response_df: pd.DataFrame,
        predictors_df: pd.DataFrame,
        perturbed_tf: str,
        feature_col: str = "target_symbol",
        feature_blacklist: list[str] | None = None,
        top_n: int | None = None,
    ):
        """
        Initialize ModelingInputData with response and predictor matrices. Note that the
        response and predictor dataframes will be subset down to the features in common
        between them, by index. The rows in both dataframes will also be ordered such
        that they match, again by index.

        :param response_df: A two column DataFrame containing the `feature_col` and
            numeric column representing the response variable.
        :param predictors_df: A Dataframe containing the `feature_col` and predictor
            numeric columns that represent the predictor variables.
        :param perturbed_tf: Name of the perturbed TF. **Note**: this must exist as a
            column in predictors_df.
        :param feature_col: Name of the column to use as the feature index. This column
            must exist in both the response and predictor DataFrames.
            (default: "target_symbol").
        :param feature_blacklist: List of feature names to exclude from analysis.
        :param top_n: If specified, retain only the top N features with the strongest
            binding scores for the perturbed TF. If this is passed on initialization,
            then the top_n_masked is set to True by default. If you wish to extract
            unmasked data, you can set `object.top_n_masked = False`. The mask can be
            toggled on and off at will.

        """
        if not isinstance(response_df, pd.DataFrame):
            raise ValueError("response_df must be a DataFrame.")
        if not isinstance(predictors_df, pd.DataFrame):
            raise ValueError("predictors_df must be a DataFrame.")
        if not isinstance(perturbed_tf, str):
            raise ValueError("perturbed_tf must be a string representing the TF name.")
        if not isinstance(feature_col, str):
            raise ValueError(
                "feature_col must be a string representing the feature name."
            )
        if feature_blacklist is not None and not isinstance(feature_blacklist, list):
            raise ValueError("feature_blacklist must be a list or None.")
        if top_n is not None and not isinstance(top_n, int):
            raise ValueError("top_n must be an integer or None.")

        self.perturbed_tf = perturbed_tf
        self.feature_col = feature_col
        self._top_n_masked = False

        # Ensure feature_blacklist is a list
        if feature_blacklist is None:
            feature_blacklist = []

        # Ensure perturbed_tf is in the blacklist
        if perturbed_tf not in feature_blacklist:
            logger.warning(
                f"Perturbed TF '{perturbed_tf}' not in blacklist. "
                f"Adding to blacklist. Setting blacklist_masked to True. "
                f"If you do not wish to blacklist the perturbed TF, "
                f"set blacklist_masked to False."
            )
            feature_blacklist.append(perturbed_tf)

        self.feature_blacklist = set(feature_blacklist)
        self.blacklist_masked = bool(self.feature_blacklist)

        # Ensure the response and predictors only contain common features
        self.response_df = response_df
        self.predictors_df = predictors_df

        # Assign top_n value
        self.top_n = top_n

    @property
    def response_df(self) -> pd.DataFrame:
        """
        Get the response DataFrame with feature masks applied.

        Returns a version of the response matrix filtered by:
        - Feature blacklist (if `blacklist_masked` is True)
        - Top-N feature selection (if `top_n_masked` is True)

        The final DataFrame will be aligned in index order with the predictors matrix.

        :return: Filtered and ordered response DataFrame.

        """
        response_df = self._response_df.copy()

        # Apply blacklist masking
        if self.blacklist_masked:
            response_df = response_df.loc[
                response_df.index.difference(self.feature_blacklist)
            ]

        # Apply top-n feature masking
        if self.top_n_masked and self.top_n_features:
            response_df = response_df.loc[self.top_n_features]
            response_df = response_df.reindex(self.predictors_df.index)

        return response_df

    @response_df.setter
    def response_df(self, value: pd.DataFrame) -> None:
        """
        Set the response DataFrame and enforce schema and consistency constraints.

        The input DataFrame must contain:
        - The feature identifier column specified by `feature_col`
        - Exactly one numeric column (excluding `feature_col`)

        After setting, the internal response matrix will use `feature_col` as its index.
        If the predictors DataFrame has already been set, both matrices will be subset
        and reordered to retain only their shared features.

        :param value: DataFrame containing response values.
        :raises ValueError: If input is not a DataFrame or does not contain
            exactly one numeric column.
        :raises KeyError: If required columns are missing.

        """
        if not isinstance(value, pd.DataFrame):
            raise ValueError("response_df must be a DataFrame.")
        if self.feature_col not in value.columns:
            raise KeyError(
                f"Feature column '{self.feature_col}' not found in response DataFrame."
            )

        # Ensure the response DataFrame has exactly one numeric
        # column (excluding feature_col)
        numeric_cols = value.drop(columns=[self.feature_col]).select_dtypes(
            include="number"
        )
        if numeric_cols.shape[1] != 1:
            raise ValueError(
                "Response DataFrame must have exactly one numeric "
                "column other than the feature_col."
            )

        logger.info(f"Response column names: {numeric_cols.columns}")

        self._response_df = value.set_index(self.feature_col)
        if hasattr(self, "_predictors_df") and self._predictors_df is not None:
            self._set_common_features_and_order()

    @property
    def predictors_df(self) -> pd.DataFrame:
        """
        Get the predictors DataFrame with feature masks applied.

        The returned DataFrame reflects any active blacklist or top-N filtering.

        """
        predictors_df = self._predictors_df.copy()

        # Apply blacklist masking
        if self.blacklist_masked:
            predictors_df = predictors_df.loc[
                predictors_df.index.difference(self.feature_blacklist)
            ]

        # Apply top-n feature masking
        if self.top_n_masked and self.top_n_features:
            predictors_df = predictors_df.loc[self.top_n_features, :]

        return predictors_df

    @predictors_df.setter
    def predictors_df(self, value: pd.DataFrame) -> None:
        """
        Set the predictors DataFrame and enforce schema constraints.

        The input DataFrame must include the `feature_col` (used as the index)
        and the column corresponding to the perturbed transcription factor.
        After setting, the response and predictor matrices will be aligned to
        retain only common features.

        :param value: DataFrame containing predictor features.
        :raises ValueError: If input is not a DataFrame.
        :raises KeyError: If required columns are missing.

        """
        if not isinstance(value, pd.DataFrame):
            raise ValueError("predictors_df must be a DataFrame.")
        if self.feature_col not in value.columns:
            raise KeyError(
                f"Feature column '{self.feature_col}' "
                "not found in predictors DataFrame."
            )
        if self.perturbed_tf not in value.columns:
            raise KeyError(
                f"Perturbed TF '{self.perturbed_tf}' not found in predictor index."
            )

        self._predictors_df = value.set_index(self.feature_col)
        if hasattr(self, "_response_df") and self._response_df is not None:
            self._set_common_features_and_order()

    @property
    def top_n(self) -> int | None:
        """
        Get the threshold for top-ranked feature selection.

        If set to an integer, this defines how many of the highest-ranked features
        (based on `predictors_df[self.perturbed_tf]`) should be retained. Ranking is
        descending (higher values rank higher). If the cutoff falls on a tie,
        fewer than N features may be selected to preserve a consistent threshold. The
        most impactful tie is when the majority of the lower ranked features have
        the same value, eg an enrichment of 0 or pvalue of 1.0.

        If set to None, top-N feature selection is disabled.

        Note: Whether top-N filtering is actively applied depends on the
        `top_n_masked` attribute. You can set `top_n_masked = False` to access the
        unfiltered data, even if `top_n` is set.

        :return: The current top-N threshold or None.

        """
        return self._top_n

    @top_n.setter
    def top_n(self, value: int | None) -> None:
        """
        Set the top-N threshold and update the feature mask.

        :param value: Positive integer or None.
        :raises ValueError: If value is not a positive integer or None.

        """
        # validate that top_n is an int greater than 0
        if value is not None and (not isinstance(value, int) or value <= 0):
            raise ValueError("top_n must be a positive integer or None.")
        # if top_n is None, set _top_n to None and _top_n_masked to False
        if value is None:
            self._top_n = value
            self._top_n_masked = False
        # else, find the top_n features according to predictors_df[perturbed_tf]
        else:
            perturbed_df = self.predictors_df.loc[:, self.perturbed_tf]

            # Rank in descending order (higher values get lower ranks)
            ranks = pd.Series(
                rankdata(-perturbed_df, method="average"),
                index=perturbed_df.index,
            )

            # Count occurrences of each unique rank
            rank_counts = ranks.value_counts().sort_index(ascending=True)
            cumulative_counts = rank_counts.cumsum()

            # Find the highest rank where cumulative count ≤ top_n
            selected_rank = (
                rank_counts.index[cumulative_counts <= value].max()
                if (cumulative_counts <= value).any()
                else None
            )

            # Subset based on the determined rank threshold
            selected_features = (
                self.predictors_df.loc[ranks <= selected_rank].index.tolist()
                if selected_rank is not None
                else []
            )

            # Store results and log info
            self._top_n = value
            self.top_n_features = selected_features
            logger.info(
                f"Selected {len(selected_features)} top features based on "
                f"descending ranking of predictors_df['{self.perturbed_tf}']."
            )
            self.top_n_masked = True

    @property
    def top_n_masked(self) -> bool:
        """
        Get the status of top-n feature masking.

        If this is `True`, then
        the top-n feature selection is applied to the predictors and response

        """
        return self._top_n_masked

    @top_n_masked.setter
    def top_n_masked(self, value: bool) -> None:
        """Set the status of top-n feature masking."""
        if not isinstance(value, bool):
            raise ValueError("top_n_masked must be a boolean.")
        if value:
            logger.info("Top-n feature masking enabled.")
        else:
            logger.info("Top-n feature masking disabled.")
        self._top_n_masked = value

    def _set_common_features_and_order(self) -> None:
        """Ensures that the response and predictor dataframes contain only the common
        features and are ordered identically based on `feature_col`."""
        # Identify common features between response_df and predictors_df
        common_feature_set = set(self._response_df.index).intersection(
            set(self._predictors_df.index)
        )

        if not common_feature_set:
            raise ValueError(
                "No common features found between response and predictors DataFrames."
            )

        logger.info(
            f"Common features between response and predictors: "
            f"{len(common_feature_set)}. "
            f"Subsetting and reordering both dataframes."
        )

        # Apply blacklist before subsetting
        if self.blacklist_masked:
            # log the intersect between the common features and the blacklist as the
            # number of blacklisted genes
            logger.info(
                f"Number of blacklisted features: "
                f"{len(common_feature_set.intersection(self.feature_blacklist))}"
            )
            common_feature_set -= self.feature_blacklist

        # Subset both dataframes based on the common features
        response_df_filtered = self._response_df.loc[list(common_feature_set)]
        predictors_df_filtered = self._predictors_df.loc[list(common_feature_set)]

        # Ensure response_df is ordered according to predictors_df
        response_df_ordered = response_df_filtered.loc[predictors_df_filtered.index]

        # raise an error if the indicies of the response and predictors
        # do not match after filtering
        if not response_df_ordered.index.equals(predictors_df_filtered.index):
            raise ValueError(
                "Indices of response_df and predictors_df do not match after "
                "filtering for common features. Please check your input data."
            )

        # Set the response and predictor DataFrames with ordered features
        self._response_df = response_df_ordered
        self._predictors_df = predictors_df_filtered

    def get_modeling_data(
        self,
        formula: str,
        add_row_max: bool = False,
        drop_intercept: bool = False,
        center_scale: bool = False,
    ) -> pd.DataFrame:
        """
        Get the predictors for modeling, optionally adding a row-wise max feature.

        :param formula: The formula to use for modeling.
        :param add_row_max: Whether to add a row-wise max feature to the predictors.
        :param drop_intercept: If `drop_intercept` is True, "-1" will be appended to
            the formula string. This will drop the intercept (constant) term from
            the model matrix output by patsy.dmatrix. Default is `False`. Note
            that if this is `False`, but `center_scale` is `True`, then the
            StandardScaler `with_mean = False` and the data is only scaled,
            not centered.
        :param center_scale: If True, apply sklearn StandardScaler after design matrix
            creation.
        :return: The design matrix for modeling. self.response_df can be used for the
            response variable.

        :raises ValueError: If the formula is not provided
        :raises PatsyError: If there is an error in creating the model matrix

        """
        if not formula:
            raise ValueError("Formula must be provided for modeling.")

        if drop_intercept:
            formula += " - 1"

        predictors_df = self.predictors_df  # Apply top-n feature mask

        # Add row-wise max feature if requested
        if add_row_max:
            predictors_df["row_max"] = predictors_df.max(axis=1)

        # Create a design matrix using patsy
        try:
            design_matrix = dmatrix(
                formula,
                data=predictors_df,
                return_type="dataframe",
                NA_action="raise",
            )
        except PatsyError as exc:
            logger.error(
                f"Error in creating model matrix with formula '{formula}': {exc}"
            )
            raise

        if center_scale:
            logger.info(f"Center matrix = `{drop_intercept}`). Scale matrix = `True`")
            scaler = StandardScaler(with_mean=drop_intercept)
            scaled_values = scaler.fit_transform(design_matrix)
            design_matrix = pd.DataFrame(
                scaled_values, index=design_matrix.index, columns=design_matrix.columns
            )

        return design_matrix

    @classmethod
    def from_files(
        cls,
        response_path: str,
        predictors_path: str,
        perturbed_tf: str,
        feature_col: str = "target_symbol",
        feature_blacklist_path: str | None = None,
        top_n: int = 600,
    ) -> "ModelingInputData":
        """
        Load response and predictor data from files. This would be considered an
        overloaded constructor in other languages. The input files must be able to be
        read into objects that satisfy the __init__ method -- see __init__ docs.

        :param response_path: Path to the response file (CSV).
        :param predictors_path: Path to the predictors file (CSV).
        :param perturbed_tf: The perturbed TF.
        :param feature_col: The column name representing features.
        :param feature_blacklist_path: Path to a file containing a list of features to
            exclude.
        :param top_n: Maximum number of features for top-n selection.
        :return: An instance of ModelingInputData.
        :raises FileNotFoundError: If the response or predictor files are missing.

        """
        if not os.path.exists(response_path):
            raise FileNotFoundError(f"Response file '{response_path}' does not exist.")
        if not os.path.exists(predictors_path):
            raise FileNotFoundError(
                f"Predictors file '{predictors_path}' does not exist."
            )

        response_df = pd.read_csv(response_path)
        predictors_df = pd.read_csv(predictors_path)

        # Load feature blacklist if provided
        feature_blacklist: list[str] = []
        if feature_blacklist_path:
            if not os.path.exists(feature_blacklist_path):
                raise FileNotFoundError(
                    f"Feature blacklist file '{feature_blacklist_path}' does not exist."
                )
            with open(feature_blacklist_path) as f:
                feature_blacklist = [line.strip() for line in f if line.strip()]

        return cls(
            response_df,
            predictors_df,
            perturbed_tf,
            feature_col,
            feature_blacklist,
            top_n,
        )


class BootstrappedModelingInputData:
    """
    This class handles bootstrapped resampling of a response vector and model matrix.

    This class supports both on-the-fly generation and externally provided bootstrap
    indices. For each bootstrap sample, it maintains sample weights derived from
    frequency counts of resampled instances.

    """

    def __init__(
        self,
        response_df: pd.DataFrame,
        model_df: pd.DataFrame,
        n_bootstraps: int | None = None,
        bootstrap_indices: list[np.ndarray] | None = None,
        normalize_sample_weights: bool = True,
    ) -> None:
        """
        Initialize bootstrapped modeling input.

        Either `n_bootstraps` or `bootstrap_indices` must be provided.

        :param response_df: Response variable.
        :param model_df: Predictor matrix.
        :param n_bootstraps: Number of bootstrap replicates to generate.
        :param bootstrap_indices: Precomputed bootstrap sample indices.

        :raises ValueError: if the inputs are invalid.

        """
        if not isinstance(response_df, pd.DataFrame):
            raise TypeError("response_df must be a DataFrame.")
        if not isinstance(model_df, pd.DataFrame):
            raise TypeError("model_df must be a DataFrame.")
        if not response_df.index.equals(model_df.index):
            raise ValueError("response_df and model_df must have the same index order.")
        if n_bootstraps and bootstrap_indices:
            raise ValueError(
                "Either `n_bootstraps` or `bootstrap_indices` "
                "must be provided, not both."
            )

        self.response_df: pd.DataFrame = response_df
        self.model_df: pd.DataFrame = model_df
        self.normalize_sample_weights = normalize_sample_weights

        # If bootstrap_indices is provided, set n_bootstraps based on its length
        if bootstrap_indices is not None:
            if not isinstance(bootstrap_indices, list) or not all(
                isinstance(indices, np.ndarray) for indices in bootstrap_indices
            ):
                raise ValueError("bootstrap_indices must be a list of numpy arrays.")
            if len(bootstrap_indices) == 0:
                raise ValueError("bootstrap_indices must not be empty.")
            self.n_bootstraps: int = len(bootstrap_indices)
        else:
            if n_bootstraps is None:
                raise ValueError(
                    "Either `n_bootstraps` or `bootstrap_indices` must be provided."
                )
            if not isinstance(n_bootstraps, int) or n_bootstraps <= 0:
                raise ValueError("n_bootstraps must be a positive integer.")
            self.n_bootstraps = n_bootstraps

        # Initialize attributes
        self._bootstrap_indices: list[np.ndarray] = []
        self._sample_weights: dict[int, np.ndarray] = {}

        # Validate or generate bootstrap indices
        if bootstrap_indices is not None:
            self.bootstrap_indices = bootstrap_indices
        else:
            self._generate_bootstrap_indices()

    @property
    def bootstrap_indices(self) -> list[np.ndarray]:
        """A list of arrays representing bootstrap sample indices."""
        return self._bootstrap_indices

    @bootstrap_indices.setter
    def bootstrap_indices(self, value: list[np.ndarray]) -> None:
        """
        Set bootstrap sample indices and compute sample weights.

        :param value: A list of arrays containing valid index values from `response_df`.

        :raises ValueError: If indices are invalid or contain out-of-range values.

        """
        if not isinstance(value, list) or not all(
            isinstance(indices, np.ndarray) for indices in value
        ):
            raise ValueError("bootstrap_indices must be a list of numpy arrays.")

        valid_indices = set(self.response_df.index)
        for i, indices in enumerate(value):
            if not set(indices).issubset(valid_indices):
                raise ValueError(
                    f"Bootstrap sample {i} contains invalid "
                    "indices not found in response_df."
                )

        self._bootstrap_indices = value
        self._compute_sample_weights()

    @property
    def normalize_sample_weights(self) -> bool:
        """
        Get the normalization status for sample weights.

        :return: True if sample weights are normalized, False otherwise.

        """
        return self._normalize_sample_weights

    @normalize_sample_weights.setter
    def normalize_sample_weights(self, value: bool) -> None:
        """
        Set the normalization status for sample weights.

        :param value: Boolean indicating whether to normalize sample weights.
        :raises ValueError: If the input is not a boolean.

        """
        if not isinstance(value, bool):
            raise ValueError("normalize_sample_weights must be a boolean.")
        logger.info(f"Sample weights normalization set to: {value}")
        self._normalize_sample_weights = value

    @property
    def sample_weights(self) -> dict[int, np.ndarray]:
        """
        Normalized sample weights corresponding to bootstrap samples.

        :return: A dictionary mapping bootstrap index to sample weights.

        """
        return self._sample_weights

    @sample_weights.setter
    def sample_weights(self, value: dict[int, np.ndarray]) -> None:
        """
        Set the sample weights directly.

        :param value: Dictionary mapping index to numpy arrays of weights.
        :raises ValueError: If the input is not a dictionary of int to ndarray.

        """
        if not isinstance(value, dict) or not all(
            isinstance(k, int) and isinstance(v, np.ndarray) for k, v in value.items()
        ):
            raise ValueError("sample_weights must be a dictionary of numpy arrays.")
        self._sample_weights = value

    def _generate_bootstrap_indices(self) -> None:
        """Generates bootstrap sample indices and sample weights."""
        y_indices: np.ndarray = self.response_df.index.to_numpy()

        self._bootstrap_indices = [
            resample(y_indices, replace=True) for _ in range(self.n_bootstraps)
        ]
        self._compute_sample_weights()

    def _compute_sample_weights(self) -> None:
        """
        Computes sample weights from existing bootstrap indices.

        :param integer: If False, the sample weights are divided by the number of
            observations in the response_df. If True, the sample weights are not divided
            by the number of observations.

        """
        y_indices: np.ndarray = self.response_df.index.to_numpy()
        sample_weights: dict[int, np.ndarray] = {}

        logger.info(
            f"Sample weights normalization method: {self.normalize_sample_weights}"
        )

        for i, sample in enumerate(self._bootstrap_indices):
            index_mapping = {label: idx for idx, label in enumerate(y_indices)}
            integer_indices = np.array([index_mapping[label] for label in sample])
            sample_counts = np.bincount(integer_indices, minlength=len(y_indices))

            if self.normalize_sample_weights:
                # note sample_counts.sum() == len(y_indicies) in this case, but
                # sample_counts.sum() seems to be more canonical
                sample_weights[i] = sample_counts / sample_counts.sum()
            else:
                sample_weights[i] = sample_counts

        self._sample_weights = sample_weights

    def get_bootstrap_sample(
        self, i: int
    ) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
        """
        Retrieves a bootstrap sample by index.

        :param i: Bootstrap sample index.
        :return: Tuple of (Y_resampled, X_resampled, sample_weights).
        :raises IndexError: If the index exceeds the number of bootstraps.

        """
        if i >= self.n_bootstraps or i < 0:
            raise IndexError(
                f"Bootstrap index {i} out of range. Max: {self.n_bootstraps - 1}"
            )

        sampled_indices = self.bootstrap_indices[i]
        sample_weights = self.get_sample_weight(i)

        return (
            self.response_df.loc[sampled_indices],
            self.model_df.loc[sampled_indices],
            sample_weights,
        )

    def get_sample_weight(self, i: int) -> np.ndarray:
        """
        Retrieves sample weights for a bootstrap sample.

        :param i: Bootstrap sample index.
        :return: Array of sample weights.

        """
        if i >= self.n_bootstraps or i < 0:
            raise IndexError(f"Sample weight index {i} out of range.")
        return self.sample_weights[i]

    def regenerate(self) -> None:
        """
        Re-generate, randomly, bootstrap samples and sample weights.

        This should be called if the response or predictors change.

        """
        self._generate_bootstrap_indices()

    def save_indices(self, filename: str) -> None:
        """
        Saves only the bootstrap indices to a JSON file.

        Saves the bootstrap indices to a JSON file. This can be used to persist the
        bootstrap indices for later use, allowing for reproducibility in analyses.

        :param filename: Path to the JSON file where the bootstrap indices will be
            saved. This will overwrite the file if it exists.

        """
        data = {
            "n_bootstraps": self.n_bootstraps,
            "bootstrap_indices": [
                indices.tolist() for indices in self._bootstrap_indices
            ],
        }
        with open(filename, "w") as f:
            json.dump(data, f)

    @classmethod
    def load_indices(cls, filename: str) -> list[np.ndarray]:
        """
        Loads bootstrap indices from a JSON file.

        :param filename: Path to the JSON file containing bootstrap indices. This likely
            was created by `save_indices()` method.
        :return: A list of numpy arrays representing the bootstrap indices.
        :raises FileNotFoundError: If the file does not exist.

        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File '{filename}' does not exist.")
        with open(filename) as f:
            data = json.load(f)

        return [np.array(indices) for indices in data["bootstrap_indices"]]

    def serialize(self, filename: str) -> None:
        """
        Saves the object as a JSON file.

        Serializes the current state of the BootstrappedModelingInputData object to a
        JSON file, including the response and model DataFrames, number of bootstraps,
        bootstrap indices, and sample weights.

        :param filename: Path to the JSON file where the object will be saved.
        :raises ValueError: If the filename is not a valid path or if the object cannot
            be serialized. This method will overwrite the file if it exists.

        """
        data = {
            "response_df": self.response_df.to_dict(orient="split"),
            "index_name": self.response_df.index.name,
            "model_df": self.model_df.to_dict(orient="split"),
            "n_bootstraps": self.n_bootstraps,
            "bootstrap_indices": [
                indices.tolist() for indices in self._bootstrap_indices
            ],
            "sample_weights": {k: v.tolist() for k, v in self._sample_weights.items()},
        }

        with open(filename, "w") as f:
            json.dump(data, f)

    @classmethod
    def deserialize(cls, filename: str):
        """
        Loads the object from a JSON file.

        :param filename: Path to the BootstrapModelingData JSON file.

        """
        with open(filename) as f:
            data = json.load(f)

        response_df = pd.DataFrame(**data["response_df"]).rename_axis(
            index=data["index_name"]
        )
        model_df = pd.DataFrame(**data["model_df"]).rename_axis(
            index=data["index_name"]
        )
        n_bootstraps = data["n_bootstraps"]

        instance = cls(response_df, model_df, n_bootstraps)

        # Restore bootstrap indices and sample weights
        instance._bootstrap_indices = [
            np.array(indices) for indices in data["bootstrap_indices"]
        ]
        instance._sample_weights = {
            int(k): np.array(v) for k, v in data["sample_weights"].items()
        }

        return instance

    def __iter__(self):
        """Resets the iterator and returns itself."""
        self._current_index = 0
        return self

    def __next__(self) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
        """
        Provides the next bootstrap sample for iteration.

        :return: Tuple of (Y_resampled, X_resampled, sample_weights).
        :raises StopIteration: When all bootstrap samples are exhausted.

        """
        if self._current_index >= self.n_bootstraps:
            raise StopIteration

        Y_resampled, X_resampled, sample_weights = self.get_bootstrap_sample(
            self._current_index
        )

        self._current_index += 1
        return Y_resampled, X_resampled, sample_weights


class BootstrapModelResults:
    """
    Encapsulates the results from bootstrapped stratified cross-validation modeling.

    This includes:
    - Confidence intervals for model coefficients across bootstrap iterations
    - Raw coefficient estimates from each iteration
    - Alpha values (regularization strengths) selected during each iteration
    - Methods for extracting statistically significant coefficients
    - Visualization utilities
    - Serialization and deserialization support

    """

    def __init__(
        self,
        ci_dict: dict[str, dict[str, tuple[float, float]]],
        bootstrap_coefs_df: pd.DataFrame,
        alpha_list: list[float],
        alpha_df: pd.DataFrame = pd.DataFrame(),
    ):
        """
        Initialize BootstrapModelResults.

        :param ci_dict: Nested dictionary mapping confidence levels to (low, high)
            confidence intervals for each coefficient.
        :param bootstrap_coefs_df: DataFrame of shape (n_bootstraps, n_features)
            containing coefficient values from each bootstrap sample.
        :param alpha_list: List of alpha values (regularization strength) selected
            during each bootstrap iteration.
        :param alpha_df: a dataframe with the columns 'bootstrap_idx', 'alpha', 'fold',
            and 'mse'

        """
        self.ci_dict = ci_dict
        self.bootstrap_coefs_df = bootstrap_coefs_df
        self.alpha_list = alpha_list
        self.alpha_df = alpha_df

    def extract_significant_coefficients(
        self, ci_level: str = "95.0", threshold: float = 0.0
    ) -> dict[str, tuple[float, float]]:
        """
        Extract coefficients that are statistically significant based on their bootstrap
        confidence intervals.

        A coefficient is considered significant if its entire confidence interval
            lies above `threshold` or below `-threshold`.

        :param ci_level: Confidence interval level (e.g., "95.0").
        :param threshold: Minimum effect size for significance.
        :return: Dictionary mapping coefficient names to their (low, high) CI bounds.

        """
        ci_dict_local = self.ci_dict.copy()

        # If CI level is not precomputed, calculate it
        if ci_level not in ci_dict_local:
            ci_level_numeric = float(ci_level)
            # log that the ci_level is not in the ci_dict
            logger.debug(
                f"Generating confidence intervals for ci level: {ci_level_numeric}"
            )
            ci_dict_local[ci_level] = {
                colname: (
                    np.percentile(
                        self.bootstrap_coefs_df[colname], (100 - ci_level_numeric) / 2
                    ),
                    np.percentile(
                        self.bootstrap_coefs_df[colname],
                        100 - (100 - ci_level_numeric) / 2,
                    ),
                )
                for colname in self.bootstrap_coefs_df.columns
            }

        # Select significant coefficients based on the confidence interval threshold
        significant_coefs_dict = {
            coef: bounds
            for coef, bounds in ci_dict_local[ci_level].items()
            if bounds[0] > threshold or bounds[1] < -threshold
        }

        # remove the following terms from ci_dict:
        keys_to_remove = [
            "bootstrap_idx",
            "final_training_score",
            "alpha",
            "left_asymptote",
            "right_asymptote",
            "Intercept",
        ]
        for key in keys_to_remove:
            significant_coefs_dict.pop(key, None)

        return significant_coefs_dict

    def visualize_significant_coefficients(
        self, ci_level: str = "95.0", threshold: float = 0.0
    ) -> plt.Figure | None:
        """
        Visualize the distribution of coefficients that are significant at the specified
        confidence level.

        :param ci_level: Confidence interval level (e.g., "95.0").
        :param threshold: Minimum absolute value for significance.
        :return: Matplotlib figure, or None if no significant coefficients are found.

        """
        significant_coefs = self.extract_significant_coefficients(ci_level, threshold)

        if not significant_coefs:
            print(
                f"No significant coefficients found for CI {ci_level} "
                "at threshold {threshold}."
            )
            return None

        # Extract relevant coefficients for plotting
        df_extracted = self.bootstrap_coefs_df[list(significant_coefs.keys())]

        # Create the boxplot
        fig = plt.figure(figsize=(10, 6))
        sns.boxplot(data=df_extracted, orient="h")
        plt.axvline(x=0, linestyle="--", color="black")
        plt.xlabel("Coefficient Values")
        plt.title(f"Coefficients with {ci_level}% CI outside ±{threshold}")

        return fig

    def serialize(self, filename: str, output_dir: str | None = None) -> None:
        """
        Save the results to disk.

        Creates two files:
        - `{filename}.json`: confidence intervals
        - `{filename}.pkl`: tuple of (bootstrap_coefs_df, alpha_list)

        :param filename: Base filename (without extension).
        :param output_dir: Optional directory to write files into. Uses current
            directory if not specified.

        :raises FileNotFoundError: If the specified directory does not exist.

        """
        # Validate that the output directory exists
        if output_dir:
            if not os.path.isdir(output_dir):
                raise FileNotFoundError(
                    f"The output directory '{output_dir}' does not exist. "
                    "Please create it before saving."
                )
            filepath_json = os.path.join(output_dir, f"{filename}.json")
            filepath_pkl = os.path.join(output_dir, f"{filename}.pkl")
        else:
            filepath_json = f"{filename}.json"
            filepath_pkl = f"{filename}.pkl"

        # Save confidence intervals as JSON
        with open(filepath_json, "w") as f:
            json.dump(self.ci_dict, f, indent=4)

        # Save DataFrame and alpha_list as a Pickle file
        with open(filepath_pkl, "wb") as f:
            pickle.dump((self.bootstrap_coefs_df, self.alpha_list), f)

    @classmethod
    def deserialize(
        cls, ci_dict_json: str, coefs_alphas_pkl: str
    ) -> "BootstrapModelResults":
        """
        Load model results from disk.

        :param ci_dict_json: Path to the JSON file with confidence intervals.
        :param coefs_alphas_pkl: Path to the Pickle file with coefficient matrix and
            alpha list.
        :return: A new BootstrapModelResults instance.
        :raises FileNotFoundError: If either file is missing.
        :raises ValueError: If the pickle file contents are invalid.

        """
        # Ensure both files exist before proceeding
        if not os.path.exists(ci_dict_json):
            raise FileNotFoundError(
                f"Confidence intervals file '{ci_dict_json}' not found."
            )
        if not os.path.exists(coefs_alphas_pkl):
            raise FileNotFoundError(f"Pickle file '{coefs_alphas_pkl}' not found.")

        # Load confidence intervals from JSON
        with open(ci_dict_json) as f:
            ci_dict = json.load(f)

        # Load DataFrame and alpha_list from Pickle
        with open(coefs_alphas_pkl, "rb") as f:
            loaded_data = pickle.load(f)

        # Validate loaded data
        if not isinstance(loaded_data, tuple) or len(loaded_data) != 2:
            raise ValueError(
                "Pickle file does not contain expected (DataFrame, list) format."
            )

        bootstrap_coefs_df, alpha_list = loaded_data

        return cls(ci_dict, bootstrap_coefs_df, alpha_list)

    @classmethod
    def from_jsonl(
        cls,
        db_path: str,
        bootstrap_results_table_name: str = "bootstrap_results",
        mse_table_name: str = "mse_path",
    ) -> "BootstrapModelResults":
        """
        Load bootstrap results from JSONL files. This is intended to be used with the
        sigmoid bootstrap results.

        :param db_path: Path to the directory containing the JSONL files for a given
            regulator
        :param bootstrap_results_table_name: Name of the JSONL file containing bootstrap
            coefficient/final model results
        :param mse_table_name: Name of the JSONL file containing fold-wise MSE results
            by bootstrap_idx/alpha
        :return: An instance of BootstrapModelResults
        :raises FileNotFoundError: If the JSONL files do not exist.

        """
        bootstrap_coef_results_path = os.path.join(
            db_path, f"{bootstrap_results_table_name}.jsonl"
        )
        mse_path = os.path.join(db_path, f"{mse_table_name}.jsonl")

        if not os.path.isfile(bootstrap_coef_results_path):
            raise FileNotFoundError(
                f"Results file not found: {bootstrap_coef_results_path}"
            )
        if not os.path.isfile(mse_path):
            raise FileNotFoundError(f"Results file not found: {mse_path}")

        results_rows = []
        with open(bootstrap_coef_results_path) as f:
            for line in f:
                try:
                    results_rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        if not results_rows:
            raise ValueError("No valid records found in the results JSONL file.")

        bootstrap_coef_results_df = pd.DataFrame(results_rows)

        # Handle optional MSE file
        mse_rows = []
        with open(mse_path) as f:
            for line in f:
                try:
                    mse_rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        alpha_df = pd.DataFrame(mse_rows) if mse_rows else pd.DataFrame()

        return cls(
            ci_dict={},
            bootstrap_coefs_df=bootstrap_coef_results_df,
            alpha_list=[],
            alpha_df=alpha_df,
        )


def stratified_cv_modeling(
    y: pd.DataFrame,
    X: pd.DataFrame,
    classes: np.ndarray,
    estimator: BaseEstimator = LassoCV(),
    skf: StratifiedKFold = StratifiedKFold(n_splits=4, shuffle=True, random_state=42),
    sample_weight: np.ndarray | None = None,
    **kwargs,
) -> BaseEstimator:
    """
    Fit a model using stratified cross-validation splits.

    This function wraps a scikit-learn estimator with user-defined stratified folds.
    While it defaults to `LassoCV`, any estimator with a `cv` attribute can be used.

    :param y: Response variable. Must be a single-column DataFrame.
    :param X: Predictor matrix. Must be a DataFrame with the same number of rows as `y`.
    :param classes: Array of class labels for stratification, typically generated by
        `stratification_classification()`.
    :param estimator: scikit-learn estimator to use for modeling. Must support `cv` as
        an attribute.
    :param skf: StratifiedKFold object to control how splits are generated.
    :param sample_weight: Optional array of per-sample weights for the estimator.
    :param kwargs: Additional arguments passed to the estimator's `fit()` method.

    :return: A fitted estimator with the best parameters determined via
        cross-validation.

    :raises ValueError: If inputs are misformatted or incompatible with the estimator.

    """
    # Validate data
    if not isinstance(y, pd.DataFrame):
        raise ValueError("The response variable y must be a DataFrame.")
    if y.shape[1] != 1:
        raise ValueError("The response variable y must be a single column DataFrame.")
    if not isinstance(X, pd.DataFrame):
        raise ValueError("The predictors X must be a DataFrame.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("The number of rows in X must match the number of rows in y.")
    if classes.size == 0 or not isinstance(classes, np.ndarray):
        raise ValueError("The classes must be a non-empty numpy array.")

    # Verify estimator has a `cv` attribute
    if not hasattr(estimator, "cv"):
        raise ValueError("The estimator must support a `cv` parameter.")

    # Initialize StratifiedKFold for stratified splits
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # default setting for shuffle is False, which means the partitioning is
        # deterministic and static. Recommendation for bootstrapping is to
        # set shuffle=True and use random_state = bootstrap_iteration in order to
        # have random, but reproducible, partitions
        folds = list(skf.split(X, classes))
        for warning in w:
            logger.debug(
                f"Warning encountered during stratified k-fold split: {warning.message}"
            )

    # Clone the estimator and set the `cv` attribute with predefined folds
    model = clone(estimator)
    model.cv = folds

    # Step 7: Fit the model using the custom cross-validation folds
    model.fit(
        X,
        y.values.ravel(),
        sample_weight=sample_weight,
    )

    return model


def bootstrap_stratified_cv_modeling(
    bootstrapped_data: BootstrappedModelingInputData,
    perturbed_tf_series: pd.Series,
    estimator: BaseEstimator = LassoCV(
        fit_intercept=True,
        max_iter=10000,
        selection="random",
        random_state=42,
        n_jobs=4,
    ),
    ci_percentiles: list[int | float] = [95.0, 99.0],
    use_sample_weight_in_cv: bool = False,
    **kwargs,
) -> BootstrapModelResults:
    """
    Perform bootstrapped stratified CV modeling and estimate confidence intervals for
    model coefficients.

    This function fits a model (e.g., LassoCV) to multiple bootstrap samples drawn
    from `BootstrappedModelingInputData`, using stratified cross-validation to
    select regularization parameters. Confidence intervals are computed across the
    bootstrapped coefficient estimates.

    :param bootstrapped_data: Bootstrapped samples of predictors and response data.
    :param perturbed_tf_series: Series of TF binding values for stratification.
    :param estimator: scikit-learn estimator. Must support `.fit()` with `sample_weight`
        and allow setting `.cv`. Default is `LassoCV`.
    :param ci_percentiles: List of confidence intervals (e.g., [95.0, 99.0]).
    :param use_sample_weight_in_cv: If True, weights from bootstrap resampling are used
        in model fitting. Defaults to False.
    :params kwargs: Additional keyword arguments. The following are supported:
        - bin_by_binding_only: Default False. If False,
            stratification is based on both binding and perturbation ranks.
            If True, only binding is used.
        - bins: Default is `[0, 8, 64, 512, np.inf]`. List of bin edges
            for stratification

    :return: A BootstrapModelResults object containing:
        - `ci_dict`: Dict of CI bounds per feature per percentile
        - `bootstrap_coefs_df`: DataFrame of coefficients across bootstrap samples
        - `alpha_list`: List of selected regularization parameters per model

    :raises ValueError: If inputs are incompatible or misformatted.
    :raises KeyError: If class assignment for stratification fails.

    """

    if not isinstance(bootstrapped_data, BootstrappedModelingInputData):
        raise ValueError(
            "bootstrapped_data must be an instance of BootstrappedModelingInputData."
        )

    if estimator is None:
        raise ValueError("An estimator must be provided. Default is LassoCV().")

    # Validate ci_percentiles
    if not isinstance(ci_percentiles, list) or not all(
        isinstance(x, (int, float)) and 0 < x < 100 for x in ci_percentiles
    ):
        raise ValueError(
            "ci_percentiles must be a list of integers or floats between 0 and 100."
        )

    if not isinstance(use_sample_weight_in_cv, bool):
        raise ValueError("use_sample_weight_in_cv must be a boolean.")
    logger.info(f"Using sample weights in CV: {use_sample_weight_in_cv}")

    # validate that the index of the response_df and model_df match
    if not bootstrapped_data.response_df.index.equals(bootstrapped_data.model_df.index):
        raise ValueError(
            "The index of the response_df must match the index of the model_df."
        )

    # validate that the index of the perturbed_tf_series matches the index of the
    # model_df
    if not bootstrapped_data.model_df.index.equals(perturbed_tf_series.index):
        raise ValueError(
            "The index of the perturbed TF series must match the index of the model_df."
        )

    # log the dimension of the response and model frames
    logger.info(f"Response frame shape: {bootstrapped_data.response_df.shape}")
    logger.info(f"Model frame shape: {bootstrapped_data.model_df.shape}")

    # log the columns of the model_df
    logger.info(f"Model frame columns: {bootstrapped_data.model_df.columns}")

    # set the binning strategy
    bin_by_binding_only = kwargs.pop("bin_by_binding_only", False)
    # log the binning strategy
    logger.info(
        "Using binning strategy: %s",
        ("binding only" if bin_by_binding_only else "binding and perturbation"),
    )

    # set the bin splits
    bins = kwargs.pop("bins", [0, 8, 64, 512, np.inf])
    # log the bins used for stratification
    logger.info(f"Using the following stratification bins: {bins}.")

    # initialize lists to store bootstrap results
    bootstrap_coefs = []
    alpha_list = []

    # shuffle = True means that the partitioning is random.
    # NOTE: In each iteration, the random state is updated to the current
    # bootstrap iteration index. This ensures that the randomization is
    # reproducible across different runs of the function, while still allowing
    # for variability in how each bootstrap sample is partitioned into train/test
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

    # Bootstrap iterations
    logger.info("Starting bootstrap modeling iterations...")
    for index, (y_resampled, x_resampled, sample_weight) in enumerate(
        bootstrapped_data
    ):
        logger.debug("Bootstrap iteration index: %d", index)

        # Set the random state for StratifiedKFold to the current index
        skf.random_state = index

        # the random_state for the estimator is used to choose among equally good
        # variables. I'm not sure how much this affects results -- we are making
        # a distribution of coefficients rather than letting sklearn choose a
        # model for us -- but it is, similar to StratifiedKFold above, randomized
        # but reproducible by setting random_state to the bootstrap iteration
        try:
            estimator.random_state = index
        except AttributeError:
            logger.warning("Estimator does not have a random_state attribute.")
            pass

        if use_sample_weight_in_cv:
            # this should be over the entire data set, since we are using the weights
            # to perform the sampling
            logger.info("Performing CV by sample weights")
            classes = stratification_classification(
                perturbed_tf_series.loc[bootstrapped_data.response_df.index].squeeze(),
                bootstrapped_data.response_df.squeeze(),
                bin_by_binding_only=bin_by_binding_only,
                bins=bins,
            )

            model_i = stratified_cv_modeling(
                bootstrapped_data.response_df,
                bootstrapped_data.model_df,
                classes=classes,
                estimator=estimator,
                skf=skf,
                sample_weight=sample_weight,
            )
        else:
            # this is performed on the resampled data
            logger.info("Performing CV by index partitioning")
            classes = stratification_classification(
                perturbed_tf_series.loc[y_resampled.index].squeeze(),
                y_resampled.squeeze(),
                bin_by_binding_only=bin_by_binding_only,
                bins=bins,
            )

            model_i = stratified_cv_modeling(
                y_resampled,
                x_resampled,
                classes=classes,
                estimator=estimator,
                skf=skf,
            )

        alpha_list.append(model_i.alpha_)
        bootstrap_coefs.append(model_i.coef_)
    # Convert bootstrap coefficients to DataFrame
    bootstrap_coefs_df = pd.DataFrame(bootstrap_coefs, columns=x_resampled.columns)

    # Compute confidence intervals
    ci_dict = {
        f"{ci}": {
            colname: (
                np.percentile(bootstrap_coefs_df[colname], (100 - ci) / 2),
                np.percentile(bootstrap_coefs_df[colname], 100 - (100 - ci) / 2),
            )
            for colname in bootstrap_coefs_df.columns
        }
        for ci in ci_percentiles
    }

    return BootstrapModelResults(
        ci_dict=ci_dict,
        bootstrap_coefs_df=bootstrap_coefs_df,
        alpha_list=alpha_list,
    )


class InteractorSignificanceResults:
    """
    Container for storing and analyzing the results of interactor significance testing.

    This class holds evaluations comparing the predictive power of interaction terms
    versus their corresponding main effects in a model, based on cross-validated R².

    Provides methods to:
    - Convert results to a DataFrame.
    - Serialize and deserialize results from disk.
    - Select final model terms by comparing interaction and main effect performance.

    """

    def __init__(self, evaluations: list[dict[str, Any]]):
        """
        Initialize the evaluations object.

        :param evaluations: A list of dictionaries containing significance test results.

        """
        self.evaluations = evaluations

    def to_dataframe(self) -> pd.DataFrame:
        """
        Return evaluations as a Pandas DataFrame.

        :return: DataFrame containing the significance test results.

        """
        return pd.DataFrame(self.evaluations)

    def serialize(self, filepath: str) -> None:
        """
        Save the evaluations to a JSON file.

        :param filepath: Path to the output JSON file.

        """
        with open(filepath, "w") as f:
            json.dump(self.evaluations, f, indent=4)

    @classmethod
    def deserialize(cls, filepath: str) -> "InteractorSignificanceResults":
        """
        Load evaluations from a JSON file.

        :param filepath: Path to the JSON file containing evaluation results.
        :return: An instance of `InteractorSignificanceResults`.

        :raises ValueError: If the JSON content is not a list.

        """
        with open(filepath) as f:
            evaluations = json.load(f)

        if not isinstance(evaluations, list):
            raise ValueError(
                f"Invalid JSON format: Expected a list, got {type(evaluations)}"
            )

        return cls(evaluations)

    def final_model(self) -> list[str]:
        """
        Select the preferred model terms based on R² comparison.

        For each interactor, compares R² of the full model (with interaction term) to
        that of a model where the interactor is replaced by its main effect. Whichever
        yields higher R² is retained.

        :return: List of selected model terms (interactor or main effect).

        """
        df = self.to_dataframe()

        if df.empty:
            return []

        # Select either the interactor or the variant based on max R²
        df["selected"] = np.where(
            df["avg_r2_interactor"] >= df["avg_r2_main_effect"],
            df["interactor"],
            df["variant"],
        )

        return df["selected"].tolist()


def evaluate_interactor_significance(
    input_data: ModelingInputData,
    stratification_classes: np.ndarray,
    model_variables: list[str],
    estimator: BaseEstimator = LinearRegression(fit_intercept=True),
) -> "InteractorSignificanceResults":
    """
    Compare predictive performance of interaction terms vs. their main effects.

    This function performs a stratified cross-validation comparison between:
    - The original model containing interaction terms (e.g., TF1:TF2)
    - A reduced model where each interactor is replaced by its corresponding
      main effect (e.g., TF2)

    R² scores are computed for both models using stratified CV. The delta in R²
    informs whether the interaction term adds predictive value.

    :param input_data: A `ModelingInputData` instance containing predictors
        and response.
    :param stratification_classes: Array of stratification labels for CV.
    :param model_variables: List of model terms, including interaction terms.
    :param estimator: A scikit-learn estimator to use for modeling. Default is
        `LinearRegression(fit_intercept=True)`.

    :return: An `InteractorSignificanceResults` instance with evaluation results.

    :raises KeyError: If a main effect is missing from the input data.

    """
    output = []

    response_df = input_data.response_df

    # Identify interaction terms (those with ":")
    interactors = [var for var in model_variables if ":" in var]

    logger.info(f"Testing the following interaction variables: {interactors}")

    # NOTE: add_row_max is set to True such that IF the formula includes row_max,
    # the column is present. However, if the formula doesn't not include row_max,
    # then that column will not be present in the model matrix.

    # Get the average R² of the original model
    avg_r2_original_model = stratified_cv_r2(
        response_df,
        input_data.get_modeling_data(" + ".join(model_variables), add_row_max=True),
        stratification_classes,
        estimator=estimator,
    )

    for interactor in interactors:
        # Extract main effect from interactor
        main_effect = interactor.split(":")[1]

        logger.debug(f"Testing interactor '{interactor}' with variant '{main_effect}'.")

        # Ensure main effect exists in predictors
        if main_effect not in input_data.predictors_df.columns:
            raise KeyError(f"Main effect '{main_effect}' not found in predictors.")

        # Define predictor sets for comparison
        predictors_with_main_effect = [
            var for var in model_variables if var != interactor
        ] + [
            main_effect
        ]  # Replace interactor with main effect

        # Get the average R² of the model with the main effect replacing one of the
        # interaction terms
        avg_r2_main_effect = stratified_cv_r2(
            response_df,
            input_data.get_modeling_data(
                " + ".join(predictors_with_main_effect), add_row_max=True
            ),
            stratification_classes,
            estimator=estimator,
        )

        # Store results
        output.append(
            {
                "interactor": interactor,
                "variant": main_effect,
                "avg_r2_interactor": avg_r2_original_model,
                "avg_r2_main_effect": avg_r2_main_effect,
                "delta_r2": avg_r2_main_effect - avg_r2_original_model,
            }
        )

    return InteractorSignificanceResults(output)
