import logging
import os
from itertools import islice

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LassoCV
from sklearn.model_selection import StratifiedKFold

# Import necessary classes and functions
from tfbpmodeling.lasso_modeling import (
    BootstrapModelResults,
    BootstrappedModelingInputData,
    bootstrap_stratified_cv_modeling,
    stratified_cv_modeling,
)
from tfbpmodeling.stratification_classification import (
    stratification_classification,
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
    Perform bootstrapped stratified CV modeling with iterative variable dropping based
    on confidence intervals, and return results at the final confidence interval.

    :param bootstrapped_data: Bootstrapped samples of predictors and response data.
    :param perturbed_tf_series: Series of TF binding values for stratification.
    :param estimator: scikit-learn estimator. Default is LassoCV.
    :param ci_percentile: Final confidence interval for results (e.g., 99.0).
    :param use_sample_weight_in_cv: Whether to use sample weights in CV.
    :param stabilization_ci_start: Starting confidence interval for stabilization (e.g.,
        50.0).
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

        for index, (y_resampled, x_resampled, sample_weight) in islice(
            enumerate(bootstrapped_data), num_samples_for_stabilization
        ):
            logger.debug(f"Bootstrap iteration index: {index}")
        # Set the random state for StratifiedKFold to the current index
        skf.random_state = i

        # the random_state for the estimator is used to choose among equally good
        # variables. I'm not sure how much this affects results -- we are making
        # a distribution of coefficients rather than letting sklearn choose a
        # model for us -- but it is, similar to StratifiedKFold above, randomized
        # but reproducible by setting random_state to the bootstrap iteration
        try:
            estimator.random_state = i
        except AttributeError:
            logger.warning("Estimator does not have a random_state attribute.")
            pass
        # by default, use the sample weights rather than the resampled data directly
        if kwargs.get("use_sample_weight_in_cv", True):
            # this should be over the entire data set, since we are using the weights
            # to perform the sampling
            logger.info("Performing CV by sample weights")
            classes = stratification_classification(
                perturbed_tf_series.loc[bootstrapped_data.response_df.index].squeeze(),
                bootstrapped_data.response_df.squeeze(),
                bin_by_binding_only=kwargs.get("bin_by_binding_only", False),
                bins=kwargs.get("bins", None),
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
                bin_by_binding_only=kwargs.get("bin_by_binding_only", False),
                bins=kwargs.get("bins", None),
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

        # Aggregate coefficients
        bootstrap_coefs_df = pd.DataFrame(
            bootstrap_coefs, columns=bootstrapped_data.model_df.columns
        )

        # Compute confidence intervals
        ci_dict = {
            colname: (
                np.percentile(bootstrap_coefs_df[colname], (100 - current_ci) / 2),
                np.percentile(
                    bootstrap_coefs_df[colname], 100 - (100 - current_ci) / 2
                ),
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
        if (
            previous_num_variables is not None
            and len(selected_variables) == previous_num_variables
        ):
            stabilized_variables = selected_variables
            logger.info(
                f"Stabilization achieved with {len(stabilized_variables)} variables"
            )
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
