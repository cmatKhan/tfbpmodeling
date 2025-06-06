import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LassoCV, LinearRegression

from tfbpmodeling.lasso_modeling import (
    BootstrapModelResults,
    BootstrappedModelingInputData,
    InteractorSignificanceResults,
    ModelingInputData,
    bootstrap_stratified_cv_modeling,
    evaluate_interactor_significance,
)
from tfbpmodeling.stratification_classification import (
    stratification_classification,
)
from tfbpmodeling.stratified_cv_r2 import stratified_cv_r2


def test_init_valid_data(sample_data):
    """Test successful initialization with valid data."""
    response_df, predictors_df = sample_data
    instance = ModelingInputData(response_df, predictors_df, perturbed_tf="TF1")

    assert isinstance(instance, ModelingInputData)
    assert instance.perturbed_tf == "TF1"
    assert instance.feature_col == "target_symbol"
    assert isinstance(instance.response_df, pd.DataFrame)
    assert isinstance(instance.predictors_df, pd.DataFrame)


def test_init_missing_feature_column():
    """Test initialization failure when feature_col is missing."""
    response_df = pd.DataFrame({"expression": [2.5, 3.2, 1.8, 4.1, 2.9]})
    predictors_df = pd.DataFrame({"TF1": [0.5, 0.8, 0.2, 0.9, 0.7]})

    with pytest.raises(KeyError):
        ModelingInputData(response_df, predictors_df, perturbed_tf="TF1")


def test_init_invalid_response_columns(sample_data):
    """Test error when response_df has incorrect column count."""
    response_df, predictors_df = sample_data
    response_df["extra_col"] = [1, 2, 3, 4, 5]  # Adding an extra column

    with pytest.raises(
        ValueError, match="Response DataFrame must have exactly one numeric column"
    ):
        ModelingInputData(response_df, predictors_df, perturbed_tf="TF1")


def test_init_perturbed_tf_not_in_predictors(sample_data):
    """Test error if perturbed TF is not in predictors."""
    response_df, predictors_df = sample_data

    with pytest.raises(
        KeyError, match="Perturbed TF 'TFX' not found in predictor index"
    ):
        ModelingInputData(response_df, predictors_df, perturbed_tf="TFX")


def test_blacklist_masking(sample_data):
    """Test that feature_blacklist correctly removes specified features."""
    response_df, predictors_df = sample_data
    blacklist = ["gene1", "gene3"]

    instance = ModelingInputData(
        response_df, predictors_df, perturbed_tf="TF1", feature_blacklist=blacklist
    )

    assert instance.blacklist_masked is True
    assert all(gene not in instance.predictors_df.index for gene in blacklist)


def test_perturbed_tf_automatically_blacklisted(sample_data):
    """Ensure the perturbed TF is automatically blacklisted if not explicitly in
    blacklist."""
    response_df, predictors_df = sample_data

    instance = ModelingInputData(response_df, predictors_df, perturbed_tf="TF1")

    assert "TF1" in instance.feature_blacklist
    assert instance.blacklist_masked is True


def test_top_n_feature_selection(sample_data):
    """Ensure top_n feature selection ranks features correctly."""
    response_df, predictors_df = sample_data

    instance = ModelingInputData(
        response_df, predictors_df, perturbed_tf="TF1", top_n=3
    )

    assert instance.top_n_masked is True
    assert len(instance.top_n_features) == 3


def test_top_n_invalid_values(sample_data):
    """Ensure invalid top_n values raise errors."""
    response_df, predictors_df = sample_data

    with pytest.raises(ValueError):
        ModelingInputData(response_df, predictors_df, perturbed_tf="TF1", top_n=-5)

    with pytest.raises(ValueError):
        ModelingInputData(
            response_df, predictors_df, perturbed_tf="TF1", top_n="abc"  # type: ignore
        )


def test_get_model_data_no_masking(modeling_input_instance):
    """Ensure get_model_data returns correct unmasked data."""
    data = modeling_input_instance.get_modeling_data(
        formula="TF1 + TF1:TF2 + TF1:TF3 - 1"
    )

    assert isinstance(data, pd.DataFrame)
    assert data.shape[1] == 3
    # assert that the names are "TF1", "TF1:TF2", "TF1:TF3"
    assert all(
        col in data.columns for col in ["TF1", "TF1:TF2", "TF1:TF3"]
    ), "Predictor columns are not as expected."


def test_get_model_data_with_top_n_masking(sample_data):
    """Ensure get_model_data applies top_n masking correctly."""
    response_df, predictors_df = sample_data

    instance = ModelingInputData(
        response_df, predictors_df, perturbed_tf="TF1", top_n=2
    )
    data = instance.get_modeling_data(formula="TF1 + TF1:TF2 + TF1:TF3 -1")

    assert len(data) == 2
    assert len(data.columns) == 3


def test_get_model_data_with_blacklist(sample_data):
    """Ensure get_model_data applies blacklist masking correctly."""
    response_df, predictors_df = sample_data
    blacklist = ["gene1", "gene3"]

    instance = ModelingInputData(
        response_df, predictors_df, perturbed_tf="TF1", feature_blacklist=blacklist
    )
    data = instance.get_modeling_data(formula="TF1 + TF2 + TF3")

    assert "gene1" not in data.index
    assert "gene3" not in data.index


def test_get_model_data_invalid_formula(sample_data):
    """Ensure an invalid formula raises an error."""
    response_df, predictors_df = sample_data

    instance = ModelingInputData(response_df, predictors_df, perturbed_tf="TF1")

    with pytest.raises(ValueError):
        instance.get_modeling_data(formula="")


def test_from_files(monkeypatch, tmp_path):
    """Ensure from_files correctly loads data."""
    response_path = tmp_path / "response.csv"
    predictors_path = tmp_path / "predictors.csv"

    response_data = pd.DataFrame(
        {
            "target_symbol": ["gene1", "gene2"],
            "expression": [2.5, 3.2],
        }
    )
    predictors_data = pd.DataFrame(
        {
            "target_symbol": ["gene1", "gene2"],
            "TF1": [0.5, 0.8],
            "TF2": [1.2, 1.5],
        }
    )

    response_data.to_csv(response_path, index=False)
    predictors_data.to_csv(predictors_path, index=False)

    instance = ModelingInputData.from_files(
        response_path=str(response_path),
        predictors_path=str(predictors_path),
        perturbed_tf="TF1",
    )

    assert isinstance(instance, ModelingInputData)
    assert instance.response_df.shape[0] == 2
    assert instance.predictors_df.shape[0] == 2


def test_initialization_random_state_none(sample_data):
    """Ensure proper initialization with valid inputs."""
    response_df, predictors_df = sample_data
    input_data = ModelingInputData(response_df, predictors_df, perturbed_tf="TF1")
    model_df = input_data.get_modeling_data(formula="TF1 + TF1:TF2 + TF1:TF3 - 1")

    boot_data = BootstrappedModelingInputData(
        input_data.response_df,
        model_df,
        n_bootstraps=5,
    )

    assert isinstance(boot_data.response_df, pd.DataFrame)
    assert isinstance(boot_data.model_df, pd.DataFrame)
    assert boot_data.n_bootstraps == 5
    assert boot_data.response_df.index.equals(boot_data.model_df.index)


def test_initialization_random_state(sample_data):
    """Ensure proper initialization with valid inputs."""
    response_df, predictors_df = sample_data
    input_data = ModelingInputData(response_df, predictors_df, perturbed_tf="TF1")
    model_df = input_data.get_modeling_data(formula="TF1 + TF1:TF2 + TF1:TF3 - 1")

    data1 = BootstrappedModelingInputData(
        input_data.response_df, model_df, n_bootstraps=5, random_state=42
    )

    data2 = BootstrappedModelingInputData(
        input_data.response_df, model_df, n_bootstraps=5, random_state=42
    )

    # data1 and data2 should have the same bootstrap indices
    assert all(
        np.array_equal(i1, i2)
        for i1, i2 in zip(data1.bootstrap_indices, data2.bootstrap_indices)
    )

    # data3 should have different bootstrap indices because of different random_state
    data3 = BootstrappedModelingInputData(
        input_data.response_df, model_df, n_bootstraps=5, random_state=100
    )

    assert not all(
        np.array_equal(i1, i2)
        for i1, i2 in zip(data3.bootstrap_indices, data2.bootstrap_indices)
    )


def test_invalid_inputs():
    """Ensure errors are raised for invalid inputs."""

    # Require that response_df and model_df are DataFrames
    with pytest.raises(TypeError):
        BootstrappedModelingInputData("not a df", pd.DataFrame(), 5)

    with pytest.raises(TypeError):
        BootstrappedModelingInputData(
            pd.DataFrame(index=["a", "b", "c"]), "not a df", 5
        )

    # require that the response_df and model_df have the same index in the same order
    with pytest.raises(IndexError):
        BootstrappedModelingInputData(
            pd.DataFrame(index=["a", "b", "c"]), pd.DataFrame(index=["a", "c", "b"]), 5
        )

    # require that n_bootstraps is a positive integer
    with pytest.raises(TypeError):
        BootstrappedModelingInputData(
            pd.DataFrame(index=["a", "b", "c"]), pd.DataFrame(index=["a", "b", "c"]), -1
        )


def test_bootstrap_sample_shape(bootstrapped_sample_data):
    """Ensure bootstrap samples maintain the correct shape."""
    sample_indices, sample_weights = bootstrapped_sample_data.get_bootstrap_sample(0)

    assert isinstance(sample_indices, np.ndarray)
    assert isinstance(sample_weights, np.ndarray)
    assert (
        len(sample_indices) == bootstrapped_sample_data.response_df.shape[0]
    )  # Same number of rows


def test_bootstrap_deterministic(sample_data):
    """Ensure setting a seed makes bootstrapping deterministic."""
    response_df, predictors_df = sample_data
    input_data = ModelingInputData(response_df, predictors_df, perturbed_tf="TF1")
    model_df = input_data.get_modeling_data(formula="TF1 + TF1:TF2 + TF1:TF3 - 1")

    np.random.seed(42)
    boot_data_1 = BootstrappedModelingInputData(
        input_data.response_df, model_df, n_bootstraps=5
    )
    np.random.seed(42)
    boot_data_2 = BootstrappedModelingInputData(
        input_data.response_df, model_df, n_bootstraps=5
    )

    for i in range(5):
        sample_indices1, sample_weights1 = boot_data_1.get_bootstrap_sample(i)
        sample_indices2, sample_weights2 = boot_data_2.get_bootstrap_sample(i)
        assert all(x == y for x, y in zip(sample_indices1, sample_indices2))
        assert all(x == y for x, y in zip(sample_weights1, sample_weights2))


def test_invalid_bootstrap_index(bootstrapped_sample_data):
    """Ensure index errors are raised for out-of-range bootstrap indices."""
    with pytest.raises(IndexError):
        bootstrapped_sample_data.get_bootstrap_sample(-1)

    with pytest.raises(IndexError):
        bootstrapped_sample_data.get_bootstrap_sample(
            bootstrapped_sample_data.n_bootstraps
        )


def test_sample_weights_sum(bootstrapped_sample_data):
    """Ensure sample weights sum to 1 for each bootstrap sample."""
    for i in range(bootstrapped_sample_data.n_bootstraps):
        weights = bootstrapped_sample_data.get_sample_weight(i)
        assert np.isclose(weights.sum(), 1, atol=1e-6)


def test_sample_weights_index_error(bootstrapped_sample_data):
    """Ensure index errors are raised for invalid weight indices."""
    with pytest.raises(IndexError):
        bootstrapped_sample_data.get_sample_weight(-1)

    with pytest.raises(IndexError):
        bootstrapped_sample_data.get_sample_weight(
            bootstrapped_sample_data.n_bootstraps
        )


def test_bootstrap_iteration(bootstrapped_sample_data):
    """Ensure iteration over bootstrap samples works as expected."""
    iterator = iter(bootstrapped_sample_data)

    for _ in range(bootstrapped_sample_data.n_bootstraps):
        sample_indices, sample_weights = next(iterator)
        assert isinstance(sample_indices, np.ndarray)
        assert isinstance(sample_weights, np.ndarray)

    with pytest.raises(StopIteration):
        next(iterator)


def test_bootstrap_regenerate(bootstrapped_sample_data):
    """Ensure reset_bootstrap_samples generates new samples."""
    old_bootstrap_indices = bootstrapped_sample_data.bootstrap_indices.copy()
    bootstrapped_sample_data.regenerate()
    new_bootstrap_indices = bootstrapped_sample_data.bootstrap_indices

    assert len(old_bootstrap_indices) == len(new_bootstrap_indices)
    assert not all(
        np.array_equal(a, b)
        for a, b in zip(old_bootstrap_indices, new_bootstrap_indices)
    )


def test_bootstrap_stratified_cv_modeling(
    random_sample_data, bootstrapped_random_sample_data
):
    perturbed_tf_series = random_sample_data.predictors_df[
        random_sample_data.perturbed_tf
    ]
    estimator = LassoCV()
    """Tests bootstrap confidence interval estimation."""
    results = bootstrap_stratified_cv_modeling(
        bootstrapped_random_sample_data,
        perturbed_tf_series,
        estimator=estimator,
        ci_percentiles=[95.0, 99.0],
        use_sample_weight_in_cv=False,
    )

    # Ensure result type
    assert isinstance(results, BootstrapModelResults)

    # Validate confidence intervals
    assert isinstance(results.ci_dict, dict)
    assert all(isinstance(v, dict) for v in results.ci_dict.values())

    # Validate bootstrap coefficients
    assert isinstance(results.bootstrap_coefs_df, pd.DataFrame)
    assert (
        results.bootstrap_coefs_df.shape[0]
        == bootstrapped_random_sample_data.n_bootstraps
    )

    # Validate alpha values
    assert isinstance(results.alpha_list, list)
    assert len(results.alpha_list) == bootstrapped_random_sample_data.n_bootstraps


def test_bootstrap_stratified_cv_modeling_with_weights(
    random_sample_data, bootstrapped_random_sample_data
):
    """Tests bootstrap confidence interval estimation."""
    perturbed_tf_series = random_sample_data.predictors_df[
        random_sample_data.perturbed_tf
    ]
    estimator = LassoCV()
    results = bootstrap_stratified_cv_modeling(
        bootstrapped_random_sample_data,
        perturbed_tf_series,
        estimator=estimator,
        ci_percentiles=[95.0, 99.0],
        use_sample_weight_in_cv=True,
    )

    # Ensure result type
    assert isinstance(results, BootstrapModelResults)

    # Validate confidence intervals
    assert isinstance(results.ci_dict, dict)
    assert all(isinstance(v, dict) for v in results.ci_dict.values())

    # Validate bootstrap coefficients
    assert isinstance(results.bootstrap_coefs_df, pd.DataFrame)
    assert (
        results.bootstrap_coefs_df.shape[0]
        == bootstrapped_random_sample_data.n_bootstraps
    )

    # Validate alpha values
    assert isinstance(results.alpha_list, list)
    assert len(results.alpha_list) == bootstrapped_random_sample_data.n_bootstraps


def test_bootstrapinputmodeldata_serialize_deserialize(
    bootstrapped_random_sample_data, tmp_path
):
    """Tests serialization and deserialization of the BootstrappedModelingInputData
    class."""

    # Create instance
    model_data = bootstrapped_random_sample_data

    # Define a temporary file path
    json_file = tmp_path / "test_bootstrap.json"

    # Serialize the object
    model_data.serialize(json_file)

    # Ensure the file was created
    assert os.path.exists(json_file)

    # Deserialize the object
    loaded_data = BootstrappedModelingInputData.deserialize(json_file)

    # Check that the restored object has the same properties
    pd.testing.assert_frame_equal(model_data.response_df, loaded_data.response_df)
    pd.testing.assert_frame_equal(model_data.model_df, loaded_data.model_df)
    assert model_data.n_bootstraps == loaded_data.n_bootstraps

    # Verify bootstrap indices
    assert len(model_data.bootstrap_indices) == len(loaded_data.bootstrap_indices)
    for orig, restored in zip(
        model_data.bootstrap_indices, loaded_data.bootstrap_indices
    ):
        np.testing.assert_array_equal(orig, restored)

    # Verify sample weights
    assert len(model_data.sample_weights) == len(loaded_data.sample_weights)
    for key in model_data.sample_weights:
        np.testing.assert_array_equal(
            model_data.sample_weights[key], loaded_data.sample_weights[key]
        )


# 2. Testing `stratified_cv_r2()`
def test_stratified_cv_r2(random_sample_data, bootstrapped_random_sample_data):
    """Tests stratified cross-validation R^2 calculation."""
    response_df = bootstrapped_random_sample_data.response_df
    model_df = bootstrapped_random_sample_data.model_df

    perturbed_tf_series = random_sample_data.predictors_df[
        random_sample_data.perturbed_tf
    ]

    classes = stratification_classification(
        perturbed_tf_series.loc[response_df.index].squeeze(),
        response_df.squeeze(),
    )

    r2_value = stratified_cv_r2(
        response_df,
        model_df,
        classes,
        estimator=LinearRegression(),
    )

    # Ensure valid R^2 output
    assert isinstance(r2_value, float)
    assert -1.0 <= r2_value <= 1.0  # R² should be within a reasonable range


# 3. Testing `evaluate_interactor_significance()`
def test_evaluate_interactor_significance(
    random_sample_data, bootstrapped_random_sample_data
):
    """Tests evaluation of interactor significance."""

    perturbed_tf_series = random_sample_data.predictors_df[
        random_sample_data.perturbed_tf
    ]
    estimator = LassoCV()

    init_results = bootstrap_stratified_cv_modeling(
        bootstrapped_random_sample_data,
        perturbed_tf_series,
        estimator=estimator,
        ci_percentiles=[95.0, 99.0],
        use_sample_weight_in_cv=True,
    )

    perturbed_tf_series = random_sample_data.predictors_df[
        random_sample_data.perturbed_tf
    ]

    classes = stratification_classification(
        perturbed_tf_series.loc[random_sample_data.response_df.index].squeeze(),
        random_sample_data.response_df.squeeze(),
    )

    results = evaluate_interactor_significance(
        random_sample_data,
        stratification_classes=classes,
        model_variables=list(init_results.extract_significant_coefficients().keys()),
    )

    # Ensure the results contain expected keys
    assert isinstance(results, InteractorSignificanceResults)
    # assert all("interactor" in entry and "avg_r2" in entry for entry in results)

    # # Validate interactor match
    # assert results[0]["interactor"] == interactor

    # Ensure R² value is within valid range
    # assert -1.0 <= results[0]["avg_r2"] <= 1.0


def test_interactor_significance_results_init(sample_evaluations):
    """Test object initialization with sample data."""
    results = InteractorSignificanceResults(sample_evaluations)

    assert isinstance(results, InteractorSignificanceResults)
    assert len(results.evaluations) == 3
    assert isinstance(results.to_dataframe(), pd.DataFrame)
    assert results.to_dataframe().shape == (3, 5)  # Ensure correct column count


def test_serialize_deserialize(sample_evaluations):
    """Test saving to and loading from JSON."""
    results = InteractorSignificanceResults(sample_evaluations)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as temp_file:
        temp_filepath = temp_file.name

    try:
        # Serialize
        results.serialize(temp_filepath)
        assert temp_filepath is not None
        assert isinstance(temp_filepath, str)

        # Check file exists and is non-empty
        with open(temp_filepath) as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == 3

        # Deserialize
        loaded_results = InteractorSignificanceResults.deserialize(temp_filepath)
        assert isinstance(loaded_results, InteractorSignificanceResults)
        assert len(loaded_results.evaluations) == 3
        assert loaded_results.to_dataframe().equals(results.to_dataframe())

    finally:
        # Cleanup
        import os

        os.remove(temp_filepath)


def test_final_model(sample_evaluations):
    """Test the final_model method, ensuring correct selection of model terms."""
    results = InteractorSignificanceResults(sample_evaluations)
    final_terms = results.final_model()

    # Expected outcome:
    # - "TF1:TF2" (since 0.85 > 0.82)
    # - "TF4" (since 0.81 > 0.78)
    # - "TF5:TF6" (since tie, keeping interactor)
    expected = ["TF1:TF2", "TF4", "TF5:TF6"]
    assert sorted(final_terms) == sorted(expected)


def test_empty_results():
    """Test behavior when initialized with empty data."""
    results = InteractorSignificanceResults([])
    assert results.to_dataframe().empty
    assert results.final_model() == []


def test_invalid_deserialize():
    """Test handling of invalid JSON file structure."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as temp_file:
        temp_filepath = temp_file.name

    try:
        # Write incorrect JSON format (dict instead of list)
        with open(temp_filepath, "w") as f:
            json.dump({"interactor": "Invalid"}, f)

        with pytest.raises(ValueError, match="Invalid JSON format"):
            InteractorSignificanceResults.deserialize(temp_filepath)

    finally:
        import os

        os.remove(temp_filepath)


def test_bootstrapmodelresult_from_jsonl(tmp_path):

    # Create dummy bootstrap results JSONL
    bootstrap_results = [
        {
            "bootstrap_idx": 0,
            "alpha": 0.1,
            "final_training_score": 0.95,
            "left_asymptote": 0.0,
            "right_asymptote": 1.0,
            "Intercept": 0.5,
            "TF1": 0.2,
            "TF2": -0.3,
        }
    ]
    mse_results = [
        {"bootstrap_idx": 0, "alpha": 0.1, "fold": 0, "mse": 0.25},
        {"bootstrap_idx": 0, "alpha": 0.1, "fold": 1, "mse": 0.20},
    ]

    # Write to temp JSONL files
    bootstrap_path = tmp_path / "bootstrap_results.jsonl"
    mse_path = tmp_path / "mse_path.jsonl"

    with open(bootstrap_path, "w") as f:
        for entry in bootstrap_results:
            f.write(json.dumps(entry) + "\n")

    with open(mse_path, "w") as f:
        for entry in mse_results:
            f.write(json.dumps(entry) + "\n")

    # Run from_jsonl
    result = BootstrapModelResults.from_jsonl(str(tmp_path))

    # Assertions
    assert isinstance(result.bootstrap_coefs_df, pd.DataFrame)
    assert not result.bootstrap_coefs_df.empty
    assert isinstance(result.alpha_df, pd.DataFrame)
    assert not result.alpha_df.empty
    assert result.alpha_list == []
    assert "TF1" in result.bootstrap_coefs_df.columns
    assert "mse" in result.alpha_df.columns
