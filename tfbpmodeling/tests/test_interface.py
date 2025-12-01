# tfbpmodeling/tests/test_interface.py

import logging
import os
from types import SimpleNamespace

import pandas as pd
import pytest

import tfbpmodeling.interface as iface


class DummyResults:
    def serialize(self, *args, **kwargs):
        pass

    def extract_significant_coefficients(self, ci_level):
        return {"TF1:geneA": 0.5}


class DummyBoot:
    def __init__(self, *args, **kwargs):
        # Add response_df attribute with a simple index for stratification
        self.response_df = pd.DataFrame(
            {"response": [1.0, 2.0, 3.0]}, index=["gene1", "gene2", "gene3"]
        )
        self.model_df = pd.DataFrame(
            {"TF1:geneA": [0.1, 0.2, 0.3]}, index=["gene1", "gene2", "gene3"]
        )


@pytest.fixture(autouse=True)
def stub_all(monkeypatch, tmp_path):
    """Stub out I/O and heavy computation for linear_perturbation_binding_modeling."""

    # 1. True for response and prediction files only, False for all other paths.
    def fake_exists(path):
        base = os.path.basename(path)
        if base in ("resp.csv", "pred.csv"):
            return True
        return False

    monkeypatch.setattr(os.path, "exists", fake_exists)

    # 2. Stub ModelingInputData.from_files
    class FakeInput:
        def __init__(self):
            # Use consistent index across all dataframes
            idx = ["gene1", "gene2", "gene3"]
            self.response_df = pd.DataFrame({"TF1": [1, 2, 3]}, index=idx)
            self.predictors_df = pd.DataFrame(
                {
                    "TF1": [1, 2, 3],
                    "geneA": [3, 2, 1],
                },
                index=idx,
            )
            self.perturbed_tf = "TF1"
            self.top_n_masked = False

        @classmethod
        def from_files(cls, **kw):
            return cls()

        def get_modeling_data(self, *a, **kw):
            return self.predictors_df

    monkeypatch.setattr(iface, "ModelingInputData", FakeInput)

    # 3. Stub BootstrappedModelingInputData
    monkeypatch.setattr(iface, "BootstrappedModelingInputData", DummyBoot)

    # 4. Stub core modeling and interactor functions
    monkeypatch.setattr(
        iface, "bootstrap_stratified_cv_modeling", lambda *a, **k: DummyResults()
    )
    monkeypatch.setattr(
        iface,
        "evaluate_interactor_significance_lassocv",
        lambda *a, **k: DummyResults(),
    )
    monkeypatch.setattr(
        iface, "evaluate_interactor_significance_linear", lambda *a, **k: DummyResults()
    )
    monkeypatch.setattr(
        iface,
        "stratification_classification",
        lambda *args, **kwargs: [0] * len(args[0]),
    )

    # 5. Stub stratified_cv_modeling (returns a dummy fitted model)
    class DummyModel:
        coef_ = [0.1, 0.2]
        intercept_ = 0.5
        alpha_ = 0.01

    monkeypatch.setattr(iface, "stratified_cv_modeling", lambda *a, **k: DummyModel())

    # 6. Stub joblib.dump (validate it receives a dict with model and metadata)
    def fake_dump(obj, filepath):
        # Validate the structure if it's the model bundle
        if isinstance(obj, dict) and "model" in obj:
            assert "feature_names" in obj
            assert "formula" in obj
            assert "perturbed_tf" in obj

    monkeypatch.setattr(iface.joblib, "dump", fake_dump)

    yield


def make_args(tmp_path):
    return SimpleNamespace(
        response_file=str(tmp_path / "resp.csv"),
        predictors_file=str(tmp_path / "pred.csv"),
        perturbed_tf="TF1",
        blacklist_file="",
        top_n=1,
        # bootstrap params
        n_bootstraps=5,
        normalize_sample_weights=False,
        random_state=0,
        scale_by_std=False,
        bins=[0, 1, 2],
        # feature options
        row_max=False,
        squared_pTF=False,
        cubic_pTF=False,
        exclude_interactor_variables=[],
        add_model_variables=[],
        ptf_main_effect=False,
        # CI & iteration
        all_data_ci_level=98.0,
        topn_ci_level=90.0,
        max_iter=100,
        iterative_dropout=False,
        stabilization_ci_start=50.0,
        stage4_lasso=False,
        stage4_topn=False,
        # system
        n_cpus=1,
        output_dir=str(tmp_path / "out_dir"),
        output_suffix="",
    )


def test_linear_workflow_logs(caplog, tmp_path):
    caplog.set_level(logging.INFO)
    args = make_args(tmp_path)
    iface.linear_perturbation_binding_modeling(args)

    log = caplog.text
    assert "estimator max_iter: 100." in log
    assert "Step 1: Preprocessing" in log
    assert "Output directory created at" in log
    assert "Step 2: Bootstrap LassoCV on all data" in log
    assert "Step 3: Bootstrap LassoCV on the significant coefficients" in log
    assert "Saving the best all data model to" in log
    assert "Step 4: Running LassoCV on topn data" in log
    assert "Step 5: Test the significance of the interactor terms" in log
