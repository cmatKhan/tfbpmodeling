import argparse
import logging
import time
from typing import Literal

from configure_logger import LogLevel, configure_logger
from tfbpmodeling.interface import (
    CustomHelpFormatter,
    add_general_arguments_to_subparsers,
    common_modeling_binning_arguments,
    common_modeling_feature_options,
    common_modeling_input_arguments,
    linear_perturbation_binding_modeling,
)

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

    common_modeling_input_arguments(linear_input_group, top_n_default=600)

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

    linear_parameters_group.add_argument(
        "--stage4_lasso",
        action="store_true",
        help="Use LassoCV-based interactor significance testing in Stage 4",
    )

    linear_parameters_group.add_argument(
        "--stage4_topn",
        action="store_true",
        help="If set, perform Stage 4 evaluation on top-n data instead of all data.",
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
