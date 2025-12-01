# tfbpmodeling

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![style](https://img.shields.io/badge/%20style-sphinx-0a507a.svg)](https://www.sphinx-doc.org/en/master/usage/index.html)
[![Pytest](https://github.com/BrentLab/tfbpmodeling/actions/workflows/ci.yml/badge.svg)](https://github.com/BrentLab/tfbpmodeling/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/BrentLab/tfbpmodeling/graph/badge.svg?token=7zBsImRmjC)](https://codecov.io/gh/BrentLab/tfbpmodeling)

A Python package for transcription factor binding and perturbation (TFBP) modeling
that analyzes relationships between transcription factor binding and gene expression
perturbations using LASSO.

NOTE: this documentation is produced by AI and hasn't had significant human revision.
Please, if you find problems or there is anything confusing or missing, open an issue
and just explain where the docs stopped being helpful.

## Quick Start

### Installation

```python
python -m pip install git+https://github.com/BrentLab/tfbpmodeling.git
```

or, for the development branch `dev`:

```python
python -m pip install git+https://github.com/BrentLab/tfbpmodeling.git@dev
```

### Basic Usage

Run the main modeling command:

```bash
python -m tfbpmodeling linear_perturbation_binding_modeling \
    --response_file response_data.csv \
    --predictors_file binding_data.csv \
    --perturbed_tf YourTF
```

## Command Line Interface

The package provides a single main command with comprehensive options for modeling
transcription factor binding and perturbation data.

### Main Command

```bash
python -m tfbpmodeling linear_perturbation_binding_modeling [OPTIONS]
```

This command executes a sequential 4-stage workflow:

1. **All Data Modeling**: Models `perturbation ~ binding` on complete dataset using
  bootstrap resampling
1. **Top-N Modeling**: Extracts significant predictors and models on top-performing
  data subset
1. **Interactor Significance**: Evaluates surviving interaction terms against
  corresponding main effects
1. **Output Generation**: Produces comprehensive results with confidence intervals
  and statistics

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--response_file` | Path to response CSV file (gene expression data) |
| `--predictors_file` | Path to predictors CSV file (binding data) |
| `--perturbed_tf` | Name of perturbed TF (must match response file column) |

### Key Options

#### Input Control

- `--blacklist_file`: Exclude specific features from analysis
- `--n_bootstraps`: Number of bootstrap samples (default: 1000)
- `--random_state`: Set seed for reproducible results
- `--top_n`: Features to retain in second modeling round (default: 600)

#### Feature Engineering

- `--row_max`: Include row maximum as predictor
- `--squared_pTF`: Include squared perturbed TF term
- `--cubic_pTF`: Include cubic perturbed TF term
- `--ptf_main_effect`: Include perturbed TF main effect
- `--exclude_interactor_variables`: Exclude variables from interaction terms
- `--add_model_variables`: Add custom variables to model

#### Model Parameters

- `--all_data_ci_level`: Confidence interval for first round (default: 98.0%)
- `--topn_ci_level`: Confidence interval for second round (default: 90.0%)
- `--max_iter`: Maximum LassoCV iterations (default: 10000)
- `--iterative_dropout`: Enable iterative variable dropout
- `--stage4_lasso`: Use LassoCV for Stage 4 significance testing

#### Data Processing

- `--normalize_sample_weights`: Normalize bootstrap weights to sum to 1
- `--scale_by_std`: Scale model matrix by standard deviation (without centering)
- `--bins`: Bin edges for data stratification (default: "0,8,12,np.inf")

#### Output Control

- `--output_dir`: Results directory (default: "./linear_perturbation_binding_modeling_results")
- `--output_suffix`: Suffix for output subdirectory naming

#### System Options

- `--n_cpus`: CPU cores for parallel processing (default: 4)
- `--log-level`: Logging verbosity (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `--log-handler`: Log output destination (console, file)

### Example Commands

#### Basic Analysis

```bash
python -m tfbpmodeling linear_perturbation_binding_modeling \
    --response_file data/expression.csv \
    --predictors_file data/binding.csv \
    --perturbed_tf YPD1
```

#### Advanced Analysis with Custom Parameters

```bash
python -m tfbpmodeling linear_perturbation_binding_modeling \
    --response_file data/expression.csv \
    --predictors_file data/binding.csv \
    --perturbed_tf YPD1 \
    --n_bootstraps 2000 \
    --top_n 500 \
    --all_data_ci_level 95.0 \
    --topn_ci_level 85.0 \
    --squared_pTF \
    --ptf_main_effect \
    --iterative_dropout \
    --stage4_lasso \
    --output_dir ./results \
    --output_suffix _custom_run \
    --n_cpus 8 \
    --random_state 42
```

#### Reproducible Analysis with Feature Engineering

```bash
poetry run python -m tfbpmodeling linear_perturbation_binding_modeling \
    --response_file data/expression.csv \
    --predictors_file data/binding.csv \
    --perturbed_tf YPD1 \
    --blacklist_file data/exclude_genes.txt \
    --random_state 12345 \
    --row_max \
    --cubic_pTF \
    --add_model_variables "red_median,green_median" \
    --exclude_interactor_variables "batch_effect" \
    --bins "0,5,10,15,np.inf" \
    --normalize_sample_weights \
    --scale_by_std
```

### Input File Formats

#### Response File (expression data)

- CSV format with genes/features as rows
- First column: gene identifiers (matching predictor file)
- Subsequent columns: sample expression values
- Must contain column matching `--perturbed_tf` argument

#### Predictors File (binding data)

- CSV format with genes/features as rows
- First column: gene identifiers (matching response file)
- Subsequent columns: binding measurements for different TFs

#### Blacklist File (optional)

- Plain text file with one feature identifier per line
- Features listed will be excluded from analysis

### Output

Results are saved in the specified output directory with subdirectories for each
analysis run. Output includes:

- Model coefficients and confidence intervals
- Bootstrap statistics and distributions
- Significance testing results
- Diagnostic plots and summaries
- Log files with detailed execution information

## Documentation

For detailed documentation and API reference, see
[https://brentlab.github.io/tfbpmodeling/](https://brentlab.github.io/tfbpmodeling/)

## Development

See [CLAUDE.md](CLAUDE.md) for development setup, testing commands, and
contribution guidelines.
