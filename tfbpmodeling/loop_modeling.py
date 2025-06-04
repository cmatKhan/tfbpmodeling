import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import StratifiedKFold
from scipy.stats import rankdata
import logging
from itertools import islice
import os

# Import necessary classes and functions
from tfbpmodeling.lasso_modeling import (
    BootstrappedModelingInputData,
    stratification_classification,
    stratified_cv_modeling,
    BootstrapModelResults,
    bootstrap_stratified_cv_modeling
)

logger = logging.getLogger(__name__)

def bootstrap_stratified_cv_loop(
    bootstrapped_data: BootstrappedModelingInputData,
    perturbed_tf_series: pd.Series,
    estimator: BaseEstimator = LassoCV(
        fit_intercept=True,
        max_iter=10000,
        selection="random",
        random_state=42,
        n_jobs=4,
    ),
    ci_percentile: float = 98.0,  # Final confidence interval
    use_sample_weight_in_cv: bool = False,
    stabilization_ci_start: float = 50.0,  # Starting CI for stabilization
    num_samples_for_stabilization: int = 500,
    output_dir: str = "",
    **kwargs,
) -> BootstrapModelResults:
    """
    Perform bootstrapped stratified CV modeling with iterative variable dropping
    based on confidence intervals, and return results at the final confidence interval.

    :param bootstrapped_data: Bootstrapped samples of predictors and response data.
    :param perturbed_tf_series: Series of TF binding values for stratification.
    :param estimator: scikit-learn estimator. Default is LassoCV.
    :param ci_percentile: Final confidence interval for results (e.g., 99.0).
    :param use_sample_weight_in_cv: Whether to use sample weights in CV.
    :param stabilization_ci_start: Starting confidence interval for stabilization (e.g., 50.0).
    :param stabilization_ci_step: Step size for increasing CI during stabilization.
    :param kwargs: Additional arguments for stratification or modeling.

    :return: A BootstrapModelResults object containing aggregated results.
    """
    current_ci = stabilization_ci_start
    previous_num_variables = None
    stabilized_variables = None
    # shuffle = True means that the partitioning is random.
    # NOTE: In each iteration, the random state is updated to the current
    # bootstrap iteration index. This ensures that the randomization is
    # reproducible across different runs of the function, while still allowing
    # for variability in how each bootstrap sample is partitioned into train/test
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    logger.info(f"Starting iterative variable dropping with CI={current_ci}")
    i = 0
    while True:
        # Perform bootstrapped modeling at the current CI
        bootstrap_coefs = []
        alpha_list = []

        for index, (y_resampled, x_resampled, sample_weight) in islice(enumerate(bootstrapped_data), num_samples_for_stabilization):
            logger.debug(f"Bootstrap iteration index: {index}")

            classes = stratification_classification(
                perturbed_tf_series.loc[y_resampled.index].squeeze(),
                y_resampled.squeeze(),
                bin_by_binding_only=kwargs.get("bin_by_binding_only", False),
                bins=kwargs.get("bins", [0, 8, 64, 512, np.inf]),
            )

            model_i = stratified_cv_modeling(
                y_resampled,
                x_resampled,
                classes=classes,
                estimator=estimator,
                skf=StratifiedKFold(n_splits=4, shuffle=True, random_state=index),
            )

            alpha_list.append(model_i.alpha_)
            bootstrap_coefs.append(model_i.coef_)

        # Aggregate coefficients
        bootstrap_coefs_df = pd.DataFrame(bootstrap_coefs, columns=bootstrapped_data.model_df.columns)

        # Compute confidence intervals
        ci_dict = {
            colname: (
                np.percentile(bootstrap_coefs_df[colname], (100 - current_ci) / 2),
                np.percentile(bootstrap_coefs_df[colname], 100 - (100 - current_ci) / 2),
            )
            for colname in bootstrap_coefs_df.columns
        }

        # Select variables within the confidence interval
        selected_variables = [
            colname
            for colname, (lower, upper) in ci_dict.items()
            if lower > 0 or upper < 0
        ]

        logger.info(f"CI={current_ci}: Selected {len(selected_variables)} variables")
        output_path = os.path.join(output_dir, f"selected_variables_ci_{i}.txt")
        with open(output_path, "w") as f:
            for var in selected_variables:
                f.write(f"{var}\n")
                
        # Check for stabilization
        if previous_num_variables is not None and len(selected_variables) == previous_num_variables:
            stabilized_variables = selected_variables
            logger.info(f"Stabilization achieved with {len(stabilized_variables)} variables")
            break

        previous_num_variables = len(selected_variables)

        # Update the bootstrapped data to include only the selected variables
        bootstrapped_data.model_df = bootstrapped_data.model_df[selected_variables]
        i += 1

    # Perform final modeling at the original confidence interval
    logger.info(f"Performing final modeling at CI={ci_percentile}")
    final_results = bootstrap_stratified_cv_modeling(
        bootstrapped_data=bootstrapped_data,
        perturbed_tf_series=perturbed_tf_series,
        estimator=estimator,
        ci_percentiles=[ci_percentile],
        use_sample_weight_in_cv=use_sample_weight_in_cv,
        **kwargs,
    )

    return final_results          