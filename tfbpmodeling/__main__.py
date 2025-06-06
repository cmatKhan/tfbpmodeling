import argparse
import fcntl
import json
import logging
import os
import shutil
import time
from typing import Literal

import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.model_selection import StratifiedKFold

from configure_logger import LogLevel, configure_logger
from tfbpmodeling.lasso_modeling import (
    BootstrapModelResults,
    BootstrappedModelingInputData,
    ModelingInputData,
    bootstrap_stratified_cv_modeling,
    evaluate_interactor_significance,
    stratification_classification,
)
from tfbpmodeling.loop_modeling import bootstrap_stratified_cv_loop
from tfbpmodeling.SigmoidModel import SigmoidModel

logger = logging.getLogger("main")


def configure_logging(
    log_level: int, handler_type: Literal["console", "file"] = "console"
) -> logging.Logger:
    """
    Configure the logging for the application.

    :param log_level: The logging level to set.
    :return: A tuple of the main and shiny loggers.

    """
    # add a timestamp to the log file name
    log_file = f"tfbpmodeling_{time.strftime('%Y%m%d-%H%M%S')}.log"
    main_logger = configure_logger(
        "main", level=log_level, handler_type=handler_type, log_file=log_file
    )
    return main_logger


# this goes along with an example in the arg parser below, showing how to
# add cmd line utilies
# def run_another_command(args: argparse.Namespace) -> None:
#     """
#     Run another command with the specified arguments.

#     :param args: The parsed command-line arguments.
#     """
#     print(f"Running another command with parameter: {args.param}")


def linear_perturbation_binding_modeling(args):
    """
    :param args: Command-line arguments containing input file paths and parameters.
    """
    if not isinstance(args.max_iter, int) or args.max_iter < 1:
        raise ValueError("The `max_iter` parameter must be a positive integer.")

    max_iter = int(args.max_iter)

    logger.info(f"estimator max_iter: {max_iter}.")

    logger.info("Step 1: Preprocessing")

    # validate input files/dirs
    if not os.path.exists(args.response_file):
        raise FileNotFoundError(f"File {args.response_file} does not exist.")
    if not os.path.exists(args.predictors_file):
        raise FileNotFoundError(f"File {args.predictors_file} does not exist.")
    if os.path.exists(args.output_dir):
        logger.warning(f"Output directory {args.output_dir} already exists.")
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f"Output directory created at {args.output_dir}")

    # the output subdir is where the output of this modeling run will be saved
    output_subdir = os.path.join(
        args.output_dir, os.path.join(args.perturbed_tf + args.output_suffix)
    )
    if os.path.exists(output_subdir):
        raise FileExistsError(
            f"Directory {output_subdir} already exists. "
            "Please specify a different `output_dir`."
        )
    else:
        os.makedirs(output_subdir, exist_ok=True)
        logger.info(f"Output subdirectory created at {output_subdir}")

    # instantiate a estimator
    # NOTE: fit_intercept is set to `true`. This means the intercept WILL BE fit
    # DO NOT add a constant vector to the predictors.
    estimator = LassoCV(
        fit_intercept=True,
        selection="random",
        n_alphas=100,
        random_state=42,
        n_jobs=args.n_cpus,
        max_iter=max_iter,
    )

    input_data = ModelingInputData.from_files(
        response_path=args.response_file,
        predictors_path=args.predictors_file,
        perturbed_tf=args.perturbed_tf,
        feature_blacklist_path=args.blacklist_file,
        top_n=args.top_n,
    )

    logger.info("Step 2: Bootstrap LassoCV on all data, full interactor model")

    # Unset the top n masking -- we want to use all the data for the first round
    # modeling
    input_data.top_n_masked = False

    # extract a list of predictor variables, which are the columns of the predictors_df
    predictor_variables = input_data.predictors_df.columns.drop(input_data.perturbed_tf)

    # drop any variables which are in args.exclude_interactor_variables
    predictor_variables = [
        var
        for var in predictor_variables
        if var not in args.exclude_interactor_variables
    ]

    # create a list of interactor terms with the perturbed_tf as the first term
    interaction_terms = [
        f"{input_data.perturbed_tf}:{var}" for var in predictor_variables
    ]
    # Construct the full interaction formula, ie perturbed_tf + perturbed_tf:other_tf1 +
    # perturbed_tf:other_tf2 + ... .
    all_data_formula = f"{input_data.perturbed_tf} + {' + '.join(interaction_terms)}"

    if args.squared_pTF:
        # if --squared_pTF is passed, then add the squared perturbed TF to the formula
        squared_term = f"I({input_data.perturbed_tf} ** 2)"
        logger.info(f"Adding squared term to model formula: {squared_term}")
        all_data_formula += f" + {squared_term}"

    # if --row_max is passed, then add "row_max" to the formula
    if args.row_max:
        logger.info("Adding `row_max` to the all data model formula")
        all_data_formula += " + row_max"

    # if --add_model_variables is passed, then add the variables to the formula
    if args.add_model_variables:
        logger.info(
            f"Adding model variables to the all data model "
            f"formula: {args.add_model_variables}"
        )
        all_data_formula += " + " + " + ".join(args.add_model_variables)

    # log the formula
    logger.info(f"All data formula for the full interactor model: {all_data_formula}")

    # create the bootstrapped data.
    bootstrapped_data_all = BootstrappedModelingInputData(
        response_df=input_data.response_df,
        model_df=input_data.get_modeling_data(
            all_data_formula,
            add_row_max=args.row_max,
            drop_intercept=args.drop_intercept,
        ),
        n_bootstraps=args.n_bootstraps,
        normalize_sample_weights=args.normalize_sample_weights,
        random_state=args.random_state,
    )

    logger.info(
        f"Running bootstrap LassoCV on all data with {args.n_bootstraps} bootstraps"
    )
    if args.iterative_dropout:
        logger.info("Using iterative dropout modeling for all data results.")
        all_data_results = bootstrap_stratified_cv_loop(
            bootstrapped_data=bootstrapped_data_all,
            perturbed_tf_series=input_data.predictors_df[input_data.perturbed_tf],
            estimator=estimator,
            ci_percentile=float(args.all_data_ci_level),
            stabilization_ci_start=args.stabilization_ci_start,
            use_sample_weight_in_cv=args.use_weights_in_cv,
            bin_by_binding_only=args.bin_by_binding_only,
            bins=args.bins,
            output_dir=output_subdir,
        )
    else:
        logger.info("Using standard bootstrap modeling for all data results.")
        all_data_results = bootstrap_stratified_cv_modeling(
            bootstrapped_data=bootstrapped_data_all,
            perturbed_tf_series=input_data.predictors_df[input_data.perturbed_tf],
            estimator=estimator,
            use_sample_weight_in_cv=args.use_weights_in_cv,
            ci_percentiles=[float(args.all_data_ci_level)],
            bin_by_binding_only=args.bin_by_binding_only,
            bins=args.bins,
        )
    # create the all data object output subdir
    all_data_output = os.path.join(output_subdir, "all_data_result_object")
    os.makedirs(all_data_output, exist_ok=True)

    logger.info(f"Serializing all data results to {all_data_output}")
    all_data_results.serialize("result_obj", all_data_output)

    # Extract the coefficients that are significant at the specified confidence level
    all_data_sig_coefs = all_data_results.extract_significant_coefficients(
        ci_level=args.all_data_ci_level,
    )

    logger.info(f"all_data_sig_coefs: {all_data_sig_coefs}")

    if not all_data_sig_coefs:
        logger.warning(
            f"No significant coefficients found at {args.all_data_ci_level}% "
            "confidence level. Exiting."
        )
        return

    # write all_data_sig_coefs to a json file
    all_data_ci_str = str(args.all_data_ci_level).replace(".", "-")
    all_data_output_file = os.path.join(
        output_subdir, f"all_data_significant_{all_data_ci_str}.json"
    )
    logger.info(f"Writing the all data significant results to {all_data_output_file}")
    with open(
        all_data_output_file,
        "w",
    ) as f:
        json.dump(all_data_sig_coefs, f, indent=4)

    logger.info(
        "Step 3: Running LassoCV on topn data with significant coefficients "
        "from the all data model"
    )

    # Create the formula for the topn modeling from the significant coefficients
    # NOTE: to remove the intercept, we need to add " -1 "
    topn_formula = f"{' + '.join(all_data_sig_coefs.keys())}"
    logger.info(f"Topn formula: {topn_formula}")

    # apply the top_n masking
    input_data.top_n_masked = True

    # Create the bootstrapped data for the topn modeling
    bootstrapped_data_top_n = BootstrappedModelingInputData(
        response_df=input_data.response_df,
        model_df=input_data.get_modeling_data(
            topn_formula, add_row_max=args.row_max, drop_intercept=args.drop_intercept
        ),
        n_bootstraps=args.n_bootstraps,
        normalize_sample_weights=args.normalize_sample_weights,
        random_state=(
            args.random_state + 10 if args.random_state else args.random_state
        ),
    )

    logger.debug(
        f"Running bootstrap LassoCV on topn data with {args.n_bootstraps} bootstraps"
    )
    topn_results = bootstrap_stratified_cv_modeling(
        bootstrapped_data_top_n,
        input_data.predictors_df[input_data.perturbed_tf],
        estimator=estimator,
        use_sample_weight_in_cv=args.use_weights_in_cv,
        ci_percentiles=[float(args.topn_ci_level)],
    )

    # create the topn data object output subdir
    topn_output = os.path.join(output_subdir, "topn_result_object")
    os.makedirs(topn_output, exist_ok=True)

    logger.info(f"Serializing topn results to {topn_output}")
    topn_results.serialize("result_obj", topn_output)

    # extract the topn_results at the specified confidence level
    topn_output_res = topn_results.extract_significant_coefficients(
        ci_level=args.topn_ci_level
    )

    logger.info(f"topn_output_res: {topn_output_res}")

    if not topn_output_res:
        logger.warning(
            f"No significant coefficients found at {args.topn_ci_level}% "
            "confidence level. Exiting."
        )
        return

    # write topn_output_res to a json file
    topn_ci_str = str(args.topn_ci_level).replace(".", "-")
    topn_output_file = os.path.join(
        output_subdir, f"topn_significant_{topn_ci_str}.json"
    )
    logger.info(f"Writing the topn significant results to {topn_output_file}")
    with open(topn_output_file, "w") as f:
        json.dump(topn_output_res, f, indent=4)

    logger.info(
        "Step 4: Test the significance of the interactor terms that survive "
        "against the corresoponding main effect"
    )

    # unmask the data
    input_data.top_n_masked = False

    # calculate the statification classes for the perturbed TF (all data)
    alldata_classes = stratification_classification(
        input_data.predictors_df[input_data.perturbed_tf].squeeze(),
        input_data.response_df.squeeze(),
        bin_by_binding_only=args.bin_by_binding_only,
        bins=args.bins,
    )

    # test the significance of the interactor against the main effect
    results = evaluate_interactor_significance(
        input_data,
        stratification_classes=alldata_classes,
        model_variables=list(
            topn_results.extract_significant_coefficients(ci_level="90.0").keys()
        ),
    )

    output_significance_file = os.path.join(
        output_subdir, "interactor_vs_main_result.json"
    )
    logger.info(
        "Writing the final interactor significance "
        "results to {output_significance_file}"
    )
    results.serialize(output_significance_file)


def create_database(
    args: argparse.Namespace,
    bootstrap_results_table_name: str = "bootstrap_results",
    mse_table_name: str = "mse_path",
):
    """
    Prepare a JSONL output directory and optionally clear an existing one.

    If `overwrite` is True, the existing directory at `args.db_path` is deleted.

    :param args: Argument namespace with 'db_path' and 'overwrite' attributes.
    :param bootstrap_results_table_name: File name for bootstrap results.
    :param mse_table_name: File name for MSE results.

    """
    if os.path.exists(args.db_path):
        if args.overwrite:
            shutil.rmtree(args.db_path)
            logger.info(f"Existing directory at {args.db_path} removed.")
        else:
            logger.info(
                f"Directory already exists at {args.db_path}. Skipping creation."
            )

    # Always recreate the directory if it was removed or didn't exist
    os.makedirs(args.db_path, exist_ok=True)
    logger.info(f"Directory {args.db_path} is ready for JSONL output.")

    # Touch the .jsonl files
    bootstrap_jsonl_path = os.path.join(
        args.db_path, f"{bootstrap_results_table_name}.jsonl"
    )
    mse_jsonl_path = os.path.join(args.db_path, f"{mse_table_name}.jsonl")

    for path in [bootstrap_jsonl_path, mse_jsonl_path]:
        open(path, "a").close()  # touch: create if doesn't exist
        logger.info(f"Initialized empty file: {path}")

    logger.info(
        f"JSONL files initialized:\n"
        f"- {bootstrap_jsonl_path}\n"
        f"- {mse_jsonl_path}"
    )


def insert_result(
    i: int,
    db_path: str,
    result_row: dict,
    max_retries: int = 50,
    retry_wait: int = 5,
):
    """
    Append a single result row to a JSONL (newline-delimited JSON) file using fcntl for
    concurrency control.

    :param i: Bootstrap index (used for jitter and logging).
    :param db_path: Path to the output JSONL file.
    :param result_row: Dictionary of results to write.
    :param max_retries: Max retries if file is locked or busy.
    :param retry_wait: Base wait time between retries.

    """
    rng = np.random.default_rng(seed=i)

    for attempt in range(max_retries):
        logger.debug(
            f"Attempting to write bootstrap {i} to {db_path} "
            f"(attempt {attempt + 1}/{max_retries})"
        )
        try:
            with open(db_path, "a") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                f.write(json.dumps(result_row) + "\n")
                f.flush()
                os.fsync(f.fileno())  # ensure it's written to disk
                fcntl.flock(f, fcntl.LOCK_UN)

            logger.debug(f"Successfully wrote bootstrap {i} to {db_path}.")
            return
        except Exception as e:
            logger.warning(f"[{i}] Write failed with error: {e}. Retrying...")
            time.sleep(retry_wait + rng.uniform(0, 2))

    logger.error(
        f"Failed to write bootstrap {i} to {db_path} after {max_retries} retries."
    )


def sigmoid_bootstrap_worker(
    args: argparse.Namespace,
    bootstrap_results_table_name: str = "bootstrap_results",
    mse_table_name: str = "mse_path",
) -> None:

    # create input data similar to perturbed_binding_modeling(). There needs to be a
    # setting to decide whether to do "all data" or "top n" modeling

    # Select bootstrap index
    i = int(args.bootstrap_idx)

    # Load input data
    input_data = ModelingInputData.from_files(
        response_path=args.response_file,
        predictors_path=args.predictors_file,
        perturbed_tf=args.perturbed_tf,
        feature_blacklist_path=args.blacklist_file,
        top_n=args.top_n,
    )

    # Determine formula
    if input_data.top_n_masked:
        res = BootstrapModelResults.from_jsonl(
            args.db_path, bootstrap_results_table_name, mse_table_name
        )
        all_data_sig_coefs = res.extract_significant_coefficients(
            ci_level=args.ci_level
        )
        logger.info("Top N Sig Coefs:" + str(all_data_sig_coefs.keys()))
        formula = " + ".join(all_data_sig_coefs.keys())

        # check is the formula is empty / there are no significant coefficients
        if formula == "":
            logger.info("No significant coefficients found for Top N Modeling...")
            return
    else:
        predictor_variables = input_data.predictors_df.columns.drop(args.perturbed_tf)
        predictor_variables = [
            var
            for var in predictor_variables
            if var not in args.exclude_interactor_variables
        ]
        interaction_terms = [
            f"{args.perturbed_tf}:{var}" for var in predictor_variables
        ]
        formula = f"{args.perturbed_tf} + {' + '.join(interaction_terms)}"

        if args.squared_pTF:
            formula += f" + I({args.perturbed_tf} ** 2)"
        if args.row_max:
            formula += " + row_max"
        if args.add_model_variables:
            formula += " + " + " + ".join(args.add_model_variables)

    logger.info(f"Model formula: {formula}")
    model_df = input_data.get_modeling_data(
        formula, add_row_max=args.row_max, drop_intercept=args.drop_intercept
    )

    bootstrap_data = BootstrappedModelingInputData(
        response_df=input_data.response_df,
        model_df=model_df,
        n_bootstraps=args.n_bootstraps,
        normalize_sample_weights=args.normalize_sample_weights,
        random_state=args.random_state,
    )

    _, _, sample_weights = bootstrap_data.get_bootstrap_sample(i)

    classes = stratification_classification(
        input_data.predictors_df[input_data.perturbed_tf].squeeze(),
        input_data.response_df.squeeze(),
        bin_by_binding_only=args.bin_by_binding_only,
        bins=args.bins,
    )

    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=i)

    folds = list(skf.split(bootstrap_data.model_df, classes))

    estimator = SigmoidModel(warm_start=args.warm_start, alphas=args.alphas, cv=folds)
    logger.info(f"Fitting model for bootstrap {args.bootstrap_idx}")
    estimator.fit(
        bootstrap_data.model_df,
        bootstrap_data.response_df.values.ravel(),
        sample_weight=sample_weights,
        minimize_options=args.minimize_options,
    )

    result_row = {
        "bootstrap_idx": i,
        "alpha": estimator.alpha_,
        "final_training_score": estimator.score(
            bootstrap_data.model_df, bootstrap_data.response_df.values.ravel()
        ),
        "left_asymptote": estimator.left_asymptote_,
        "right_asymptote": estimator.right_asymptote_,
        **dict(zip(bootstrap_data.model_df.columns, estimator.coef_)),
    }

    # this output dir will store results from all_data step separate from top_n step
    if input_data.top_n_masked:
        output_root = os.path.join(args.db_path, f"_top_{args.top_n}")
        # create directory if it doesn't already exist
        os.makedirs(output_root, exist_ok=True)
    else:
        output_root = args.db_path

    insert_result(
        i,
        os.path.join(output_root, f"{bootstrap_results_table_name}.jsonl"),
        result_row,
    )

    # Save MSE path if present
    if hasattr(estimator, "mse_path_") and hasattr(estimator, "alphas_"):
        n_alphas, n_folds = estimator.mse_path_.shape
        for a_idx in range(n_alphas):
            for f_idx in range(n_folds):
                mse_row = {
                    "bootstrap_idx": i,
                    "alpha": estimator.alphas_[a_idx],
                    "fold": f_idx,
                    "mse": estimator.mse_path_[a_idx, f_idx],
                }
                insert_result(
                    i, os.path.join(output_root, f"{mse_table_name}.jsonl"), mse_row
                )

    logger.info(f"Completed bootstrap {i}")


def test_sigmoid_interactor_significance(
    args: argparse.Namespace,
    bootstrap_results_table_name: str = "bootstrap_results",
    mse_table_name: str = "mse_path",
) -> None:
    """
    Test the significance of interactor terms against the main effect.

    :param args: Command-line arguments containing input file paths and parameters.
    :param bootstrap_results_table_name: File name for bootstrap results.
    :param mse_table_name: File name for MSE results.

    """
    # check that the topn modeling output dir exists
    if not os.path.isdir(args.db_path):
        raise FileNotFoundError(
            f"Directory {args.db_path} does not exist. "
            "Please run the linear_perturbation_binding_modeling command first."
        )

    logger.info("Testing interactor significance...")

    # Load input data
    input_data = ModelingInputData.from_files(
        response_path=args.response_file,
        predictors_path=args.predictors_file,
        perturbed_tf=args.perturbed_tf,
        feature_blacklist_path=args.blacklist_file,
        top_n=args.top_n,
    )

    classes = stratification_classification(
        input_data.predictors_df[input_data.perturbed_tf].squeeze(),
        input_data.response_df.squeeze(),
        bin_by_binding_only=args.bin_by_binding_only,
        bins=args.bins,
    )

    # parse results from the previous step
    res = BootstrapModelResults.from_jsonl(
        args.db_path, bootstrap_results_table_name, mse_table_name
    )

    topn_sig_coefs = res.extract_significant_coefficients(ci_level=args.ci_level)
    logger.info("Top N Significant Coefs:" + str(topn_sig_coefs.keys()))

    model_vars = topn_sig_coefs.keys()

    results = evaluate_interactor_significance(
        input_data,
        classes,
        list(model_vars),
        SigmoidModel(),
    )

    output_significance_file = os.path.join(
        args.db_path, "interactor_vs_main_result.json"
    )
    logger.info(
        "Writing the final interactor significance "
        "results to {output_significance_file}"
    )
    results.serialize(output_significance_file)


class CustomHelpFormatter(argparse.HelpFormatter):
    """
    This could be used to customize the help message formatting for the argparse parser.

    Left as a placeholder.

    """


def parse_bins(s):
    try:
        return [np.inf if x == "np.inf" else int(x) for x in s.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid bin value in '{s}'")


def parse_comma_separated_list(value):
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_json_dict(s):
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"Invalid JSON: {e}")


# Allowed keys for method='L-BFGS-B' (excluding deprecated options)
LBFGSB_ALLOWED_KEYS = {
    "maxcor",  # int
    "ftol",  # float
    "gtol",  # float
    "eps",  # float or ndarray
    "maxfun",  # int
    "maxiter",  # int
    "maxls",  # int
    "finite_diff_rel_step",  # float or array-like or None
}


def parse_lbfgsb_options(s):
    try:
        opts = json.loads(s)
        if not isinstance(opts, dict):
            raise ValueError("Options must be a JSON object")

        unexpected_keys = set(opts) - LBFGSB_ALLOWED_KEYS
        if unexpected_keys:
            raise argparse.ArgumentTypeError(
                f"Unexpected keys in --minimize_options: {unexpected_keys}"
            )
        return opts
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"Invalid JSON: {e}")
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))


def add_general_arguments_to_subparsers(subparsers, general_arguments):
    for subparser in subparsers.choices.values():
        for arg in general_arguments:
            subparser._add_action(arg)


def common_modeling_binning_arguments(parser: argparse._ArgumentGroup) -> None:
    parser.add_argument(
        "--bins",
        type=parse_bins,
        default="0,8,64,512,np.inf",
        help=(
            "Comma-separated list of bin edges (integers or 'np.inf'). "
            "Default is --bins 0,8,12,np.inf"
        ),
    )
    parser.add_argument(
        "--bin_by_binding_only",
        action="store_true",
        help=(
            "When creating stratification classes, use binding data only instead of "
            "both binding and perturbation data. The default is to use both."
        ),
    )


def common_modeling_input_arguments(parser: argparse._ArgumentGroup) -> None:
    """Add common input arguments for modeling commands."""
    parser.add_argument(
        "--response_file",
        type=str,
        required=True,
        help=(
            "Path to the response CSV file. The first column must contain "
            "feature names or locus tags (e.g., gene symbols), matching the index "
            "format in both response and predictor files. The perturbed gene will "
            "be removed from the model data only if its column names match the "
            "index format."
        ),
    )
    parser.add_argument(
        "--predictors_file",
        type=str,
        required=True,
        help=(
            "Path to the predictors CSV file. The first column must contain "
            "feature names or locus tags (e.g., gene symbols), ensuring consistency "
            "between response and predictor files."
        ),
    )
    parser.add_argument(
        "--perturbed_tf",
        type=str,
        required=True,
        help=(
            "Name of the perturbed transcription factor (TF) used as the "
            "response variable. It must match a column in the response file."
        ),
    )
    parser.add_argument(
        "--blacklist_file",
        type=str,
        default="",
        help=(
            "Optional file containing a list of features (one per line) to be excluded "
            "from the analysis."
        ),
    )
    parser.add_argument(
        "--n_bootstraps",
        type=int,
        default=1000,
        help="Number of bootstrap samples to generate for resampling. Default is 1000",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=None,
        help="Set this to an integer to make the bootstrap sampling reproducible. "
        "Default is None (no fixed seed) and each call will produce different "
        "bootstrap indices. Note that if this is set, the `top_n` random_state will "
        "be +10 in order to make the top_n indicies different from the `all_data` step",
    )
    parser.add_argument(
        "--normalize_sample_weights",
        action="store_true",
        help=(
            "Set this to normalize the sample weights to sum to 1. " "Default is False."
        ),
    )


def common_modeling_feature_options(parser: argparse._ArgumentGroup) -> None:
    parser.add_argument(
        "--drop_intercept",
        action="store_true",
        help="Drop the intercept from the model. Default is False",
    )
    parser.add_argument(
        "--row_max",
        action="store_true",
        help=(
            "Include the row max as an additional predictor in the model matrix "
            "in the first round (all data) model."
        ),
    )
    parser.add_argument(
        "--squared_pTF",
        action="store_true",
        help=(
            "Include the squared pTF as an additional predictor in the model matrix "
            "in the first round (all data) model."
        ),
    )
    parser.add_argument(
        "--exclude_interactor_variables",
        type=parse_comma_separated_list,
        default=[],
        help=(
            "Comma-separated list of variables to exclude from the interactor terms. "
            "E.g. red_median,green_median"
        ),
    )
    parser.add_argument(
        "--add_model_variables",
        type=parse_comma_separated_list,
        default=[],
        help=(
            "Comma-separated list of variables to add to the all_data model. "
            "E.g., red_median,green_median would be added as ... + red_median + "
            "green_median"
        ),
    )


def main() -> None:
    """Main entry point for the tfbpmodeling application."""
    parser = argparse.ArgumentParser(
        prog="tfbpmodeling",
        description="tfbpmodeling Main Entry Point",
        usage="tfbpmodeling --help",
        formatter_class=CustomHelpFormatter,
    )

    formatter = parser._get_formatter()

    # Shared parameter for logging level
    log_level_argument = parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )
    log_handler_argument = parser.add_argument(
        "--log-handler",
        type=str,
        default="console",
        choices=["console", "file"],
        help="Set the logging handler",
    )
    formatter.add_arguments([log_level_argument, log_handler_argument])

    # Define subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # An example of adding another command
    # another_parser = subparsers.add_parser(
    #     "another_command",
    #     help="Run another command",
    #     description="Run another command",
    #     formatter_class=CustomHelpFormatter,
    # )
    # another_parser.add_argument(
    #     "--param", type=str, required=True, help="A parameter for another command"
    # )
    # another_parser.set_defaults(func=run_another_command)

    # Lasso Bootstrap command
    linear_lasso_parser = subparsers.add_parser(
        "linear_perturbation_binding_modeling",
        help="Run LassoCV or GeneralizedLogisticModel with bootstrap resampling",
        description=(
            "This executes the sequential workflow which models first  "
            "`perturbation ~ binding` on all of the data, then extracts the "
            "significant predictors and does the same thing on the `top n` data. "
            "Finally it evaluates the surviving interactor terms against the "
            "corresponding main effect."
        ),
        formatter_class=CustomHelpFormatter,
    )

    # Input arguments
    linear_input_group = linear_lasso_parser.add_argument_group("Input")

    common_modeling_input_arguments(linear_input_group)

    linear_model_feature_options_group = linear_lasso_parser.add_argument_group(
        "Feature Options"
    )

    common_modeling_feature_options(linear_model_feature_options_group)

    linear_model_binning_group = linear_lasso_parser.add_argument_group(
        "Binning Options"
    )
    common_modeling_binning_arguments(linear_model_binning_group)

    linear_parameters_group = linear_lasso_parser.add_argument_group("Parameters")

    linear_parameters_group.add_argument(
        "--top_n",
        type=int,
        default=600,
        help=(
            "Number of features to retain in the second round of modeling. "
            "Default is 600"
        ),
    )

    linear_parameters_group.add_argument(
        "--all_data_ci_level",
        type=float,
        default=98.0,
        help=(
            "Confidence interval threshold (in percent) for selecting significant "
            "coefficients. Default is 98.0"
        ),
    )

    linear_parameters_group.add_argument(
        "--topn_ci_level",
        type=float,
        default=90.0,
        help=(
            "Confidence interval threshold for the second round of modeling. "
            "Default is 90.0"
        ),
    )

    linear_parameters_group.add_argument(
        "--max_iter",
        type=int,
        default=10000,
        help=(
            "This controls the maximum number of iterations LassoCV may "
            "use in order to fit"
        ),
    )

    linear_parameters_group.add_argument(
        "--use_weights_in_cv",
        action="store_true",
        help=(
            "Enable sample weighting in cross-validation based on bootstrap "
            "sample proportions."
        ),
    )

    linear_parameters_group.add_argument(
        "--iterative_dropout",
        action="store_true",
        help="Enable iterative variable dropout based on confidence intervals.",
    )

    linear_parameters_group.add_argument(
        "--stabilization_ci_start",
        type=float,
        default=50.0,
        help="Starting confidence interval for iterative dropout stabilization",
    )

    # Output arguments
    linear_output_group = linear_lasso_parser.add_argument_group("Output")

    linear_output_group.add_argument(
        "--output_dir",
        type=str,
        default="./linear_perturbation_binding_modeling_results",
        help=(
            "Directory where model results will be saved. A new subdirectory "
            "is created per run."
        ),
    )

    linear_output_group.add_argument(
        "--output_suffix",
        type=str,
        default="",
        help=(
            "The subdirectory will be named by the perturbed_tf. "
            "Use output_suffix to add a suffix to the subdirectory name."
        ),
    )

    linear_system_group = linear_lasso_parser.add_argument_group("System")

    linear_system_group.add_argument(
        "--n_cpus",
        type=int,
        default=4,
        help=(
            "Number of CPUs to use for parallel processing each lassoCV call. "
            "Recommended 4"
        ),
    )

    linear_lasso_parser.set_defaults(func=linear_perturbation_binding_modeling)

    # Sigmoid worker cmds
    sigmoid_parser = subparsers.add_parser(
        "sigmoid_bootstrap_worker",
        help="Run a single bootstrap iteration of the sigmoid model",
        description=(
            "This executes a single bootstrap iteration of the sigmoid model."
        ),
        formatter_class=CustomHelpFormatter,
    )

    sigmoid_input_group = sigmoid_parser.add_argument_group("Input")

    common_modeling_input_arguments(sigmoid_input_group)

    sigmoid_input_group.add_argument(
        "--bootstrap_idx",
        type=int,
        required=True,
        help=(
            "Bootstrap index to use for the current iteration. This should be "
            "an integer corresponding to the bootstrap sample."
        ),
    )

    sigmoid_parameters_group = sigmoid_parser.add_argument_group("Parameters")

    sigmoid_parameters_group.add_argument(
        "--top_n",
        type=int,
        default=None,
        help=(
            "This is the number of features to use for second round modeling. "
            "Defaults to `Non`, for the 'all_data' model. Set to eg 600 for top_n "
            "modeling"
        ),
    )
    sigmoid_parameters_group.add_argument(
        "--ci_level",
        type=float,
        default=98.0,
        help=(
            "Confidence interval threshold for the second round of modeling. "
            "Default is 98.0. Only applied if `--top_n` is set"
        ),
    )
    sigmoid_parameters_group.add_argument(
        "--warm_start",
        action="store_true",
        help=("Enable warm start for the model. Default is False"),
    )
    sigmoid_parameters_group.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.1, 1.0, 10.0],
        help=(
            "List of alpha values to use for the model. " "Default is [0.1, 1.0, 10.0]"
        ),
    )

    sigmoid_parameters_group.add_argument(
        "--minimize_options",
        type=parse_lbfgsb_options,
        help=(
            "JSON string of options for scipy.optimize.minimize with "
            "method='L-BFGS-B'. Allowed keys (with defaults): "
            "maxcor=10, ftol=2.22e-9, gtol=1e-5, eps=1e-8, maxfun=15000, "
            "maxiter=15000, maxls=20, finite_diff_rel_step=None. "
            'Example: \'{"maxiter": 1000, "gtol": 1e-6}\''
        ),
    )

    sigmoid_parameters_group.add_argument(
        "--test_interactor_variables",
        action="store_true",
        help=(
            "If set, run last step to evaluate interactor terms against main effects."
        ),
    )

    sigmoid_model_feature_options_group = sigmoid_parser.add_argument_group(
        "Feature Options"
    )

    common_modeling_feature_options(sigmoid_model_feature_options_group)

    sigmoid_model_binning_group = sigmoid_parser.add_argument_group("Binning Options")

    common_modeling_binning_arguments(sigmoid_model_binning_group)

    sigmoid_output_group = sigmoid_parser.add_argument_group("Output")

    sigmoid_output_group.add_argument(
        "--db_path",
        type=str,
        required=True,
        help=("Path to the database file where the results will be stored."),
    )

    sigmoid_parser.set_defaults(func=sigmoid_bootstrap_worker)

    # Sigmoid worker cmds
    sigmoid_step3_parser = subparsers.add_parser(
        "sigmoid_interactor_significance",
        help="Run the interactor significance evaluation step on a sigmoid model",
        description=(
            "This executes the interactor significance evaluation step "
            "on a sigmoid model. It evaluates the interactor terms against "
            "the corresponding main effect."
        ),
        formatter_class=CustomHelpFormatter,
    )

    sigmoid_step3_input_group = sigmoid_step3_parser.add_argument_group("Input")

    common_modeling_input_arguments(sigmoid_step3_input_group)

    sigmoid_step3_input_group.add_argument(
        "--db_path",
        type=str,
        required=True,
        help=(
            "Path to the database file where the results from the previous "
            "steps are stored. This should point to the directory containing "
            "the bootstrap results."
        ),
    )
    sigmoid_step3_parameters_group = sigmoid_step3_parser.add_argument_group(
        "Parameters"
    )
    sigmoid_step3_parameters_group.add_argument(
        "--top_n",
        type=int,
        default=600,
        help=(
            "Number of features to retain in the second round of modeling. "
            "Default is 600"
        ),
    )

    sigmoid_step3_model_binning_group = sigmoid_step3_parser.add_argument_group(
        "Binning Options"
    )
    common_modeling_binning_arguments(sigmoid_step3_model_binning_group)

    sigmoid_step3_parser.set_defaults(func=test_sigmoid_interactor_significance)

    # add create_database command
    create_db_parser = subparsers.add_parser(
        "create_database",
        help="Create an database file (sqlite or csv)",
        description=(
            "Create an empty database file with WAL mode and busy timeout " "settings."
        ),
        formatter_class=CustomHelpFormatter,
    )

    create_db_parser.add_argument(
        "--db_path",
        type=str,
        required=True,
        help="Path to the database file to create.",
    )

    create_db_parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Overwrite the existing database file if it exists. " "Default is False."
        ),
    )

    create_db_parser.set_defaults(func=create_database)

    # Add the general arguments to the subcommand parsers
    add_general_arguments_to_subparsers(subparsers, [log_level_argument])

    # Parse arguments
    args = parser.parse_args()

    # Configure logging
    try:
        log_level = LogLevel.from_string(args.log_level)
    except ValueError as e:
        print(e)
        parser.print_help()
        return

    _ = configure_logging(log_level)

    # Run the appropriate command
    if args.command is None:
        parser.print_help()
    else:
        args.func(args)


if __name__ == "__main__":
    main()
