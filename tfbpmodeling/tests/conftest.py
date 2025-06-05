from typing import Any

import numpy as np
import pandas as pd
import pytest

from tfbpmodeling.lasso_modeling import (
    BootstrappedModelingInputData,
    ModelingInputData,
)


@pytest.fixture
def sample_data():
    """Fixture to create sample response and predictor data."""
    response_data = pd.DataFrame(
        {
            "target_symbol": ["gene1", "gene2", "gene3", "gene4", "gene5"],
            "expression": [2.5, 3.2, 1.8, 4.1, 2.9],
        }
    )

    predictors_data = pd.DataFrame(
        {
            "target_symbol": ["gene1", "gene2", "gene3", "gene4", "gene5"],
            "TF1": [0.5, 0.8, 0.2, 0.9, 0.7],
            "TF2": [1.2, 1.5, 1.1, 1.7, 1.3],
            "TF3": [0.3, 0.4, 0.2, 0.5, 0.3],
        }
    )

    return response_data, predictors_data


@pytest.fixture
def bootstrapped_sample_data(sample_data):
    response_df, predictors_df = sample_data
    input_data = ModelingInputData(response_df, predictors_df, perturbed_tf="TF1")
    model_df = input_data.get_modeling_data(formula="TF1 + TF1:TF2 + TF1:TF3 - 1")

    return BootstrappedModelingInputData(
        input_data.response_df, model_df, n_bootstraps=5, random_state=42
    )


@pytest.fixture
def modeling_input_instance(sample_data):
    """Creates an instance of ModelingInputData with sample data."""
    response_df, predictors_df = sample_data
    return ModelingInputData(response_df, predictors_df, perturbed_tf="TF1")


@pytest.fixture
def random_sample_data():
    """Generates synthetic response and predictor data with consistent indexing,
    ensuring the response is derived from a subset of predictors."""
    np.random.seed(42)  # Ensure reproducibility
    total_features = 100

    # Feature column names: gene1, gene2, ..., gene100
    feature_col = [f"gene{i+1}" for i in range(total_features)]

    # Select predictor columns: gene5 to gene14
    predictor_columns = feature_col[4:14]  # (gene5 to gene14)

    # Number of samples
    n_samples = 100

    # Generate predictors DataFrame
    predictors_df = pd.DataFrame(
        np.random.randn(n_samples, len(predictor_columns)),
        columns=predictor_columns,
        index=[f"gene{i+1}" for i in range(n_samples)],
    ).reset_index(names="target_symbol")

    # Generate a response variable based on gene5, gene5:gene7, and gene5:gene10
    response_values = (
        1.5 * predictors_df["gene5"]  # Main effect of gene5
        + 0.8
        * (predictors_df["gene5"] * predictors_df["gene7"])  # Interaction: gene5:gene7
        + 0.6
        * (
            predictors_df["gene5"] * predictors_df["gene10"]
        )  # Interaction: gene5:gene10
        + np.random.randn(n_samples) * 0.5  # Add noise
    )

    # Create response DataFrame
    response_df = pd.DataFrame(
        {"response": response_values, "target_symbol": predictors_df.target_symbol}
    )

    return ModelingInputData(
        response_df=response_df,
        predictors_df=predictors_df,
        perturbed_tf="gene5",
        feature_col="target_symbol",
        feature_blacklist=["gene1", "gene2"],
        top_n=20,
    )


@pytest.fixture
def bootstrapped_random_sample_data(random_sample_data):
    """Generates a BootstrappedModelingInputData instance with random sample data."""

    random_sample_data.top_n_masked = False

    # Extract predictor variables and drop the perturbed TF
    predictor_variables = random_sample_data.predictors_df.columns.drop(
        random_sample_data.perturbed_tf
    )

    # Create interaction
    interaction_terms = [
        f"{random_sample_data.perturbed_tf}:{var}" for var in predictor_variables
    ]

    # Construct the formula as a single expression (no intercept)
    formula = f"{random_sample_data.perturbed_tf} + {' + '.join(interaction_terms)} - 1"

    return BootstrappedModelingInputData(
        random_sample_data.response_df,
        random_sample_data.get_modeling_data(formula=formula),
        n_bootstraps=100,
        random_state=42,
    )


@pytest.fixture
def sample_evaluations() -> list[dict[str, Any]]:
    """Provides sample evaluation data for testing."""
    return [
        {
            "interactor": "TF1:TF2",
            "variant": "TF2",
            "avg_r2_interactor": 0.85,
            "avg_r2_main_effect": 0.82,
            "delta_r2": -0.03,
        },
        {
            "interactor": "TF3:TF4",
            "variant": "TF4",
            "avg_r2_interactor": 0.78,
            "avg_r2_main_effect": 0.81,
            "delta_r2": 0.03,
        },
        {
            "interactor": "TF5:TF6",
            "variant": "TF6",
            "avg_r2_interactor": 0.90,
            "avg_r2_main_effect": 0.90,
            "delta_r2": 0.00,
        },
    ]
