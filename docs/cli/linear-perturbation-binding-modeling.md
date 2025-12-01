# linear_perturbation_binding_modeling

The main command for running transcription factor binding and perturbation analysis.

## Synopsis

```bash
python -m tfbpmodeling linear_perturbation_binding_modeling \
    --response_file RESPONSE_FILE \
    --predictors_file PREDICTORS_FILE \
    --perturbed_tf PERTURBED_TF \
    [OPTIONS]
```

## Description

This command executes a sequential 4-stage workflow that models the relationship between transcription factor binding data and gene expression perturbation data:

1. **All Data Modeling**: Bootstrap resampling with LassoCV on complete dataset
2. **Top-N Modeling**: Secondary modeling on significant predictors from top-performing data subset
3. **Interactor Significance**: Statistical evaluation of interaction terms vs corresponding main effects
4. **Results Generation**: Comprehensive output with confidence intervals and diagnostic information

The workflow uses bootstrap resampling to provide robust statistical inference and regularized regression (LassoCV) for feature selection and model fitting.

## Required Arguments

### Input Files

#### `--response_file RESPONSE_FILE`
Path to the response CSV file containing gene expression data.

**Format Requirements:**
- CSV format with genes/features as rows
- First column: feature names or locus tags (e.g., gene symbols)
- Subsequent columns: expression values for each sample
- Must contain a column matching the `--perturbed_tf` parameter
- Gene identifiers must match those in the predictors file

**Example:**
```csv
gene_id,sample1,sample2,sample3,sample4
YPD1,0.23,-1.45,0.87,-0.12
YBR123W,1.34,0.56,-0.23,0.78
YCR456X,-0.45,0.12,1.23,-0.56
```

#### `--predictors_file PREDICTORS_FILE`
Path to the predictors CSV file containing transcription factor binding data.

**Format Requirements:**
- CSV format with genes/features as rows
- First column: feature names or locus tags matching response file
- Subsequent columns: binding measurements for different transcription factors
- Numeric values representing binding strength/probability

**Example:**
```csv
gene_id,TF1,TF2,TF3,TF4
YPD1,0.34,0.12,0.78,0.01
YBR123W,0.89,0.45,0.23,0.67
YCR456X,0.12,0.78,0.34,0.90
```

#### `--perturbed_tf PERTURBED_TF`
Name of the perturbed transcription factor used as the response variable.

**Requirements:**
- Must exactly match a column name in the response file
- Case-sensitive
- Will be used as the dependent variable in modeling

## Optional Arguments

### Input Control

#### `--blacklist_file BLACKLIST_FILE`
Optional file containing features to exclude from analysis.

**Format**: Plain text file with one feature identifier per line
**Example:**
```
YBR999W
YCR888X
control_gene
batch_effect_gene
```

#### `--n_bootstraps N_BOOTSTRAPS`
Number of bootstrap samples for resampling analysis.
- **Default**: 1000
- **Range**: 100-10000 recommended
- **Impact**: Higher values provide more robust statistics but increase runtime

#### `--random_state RANDOM_STATE`
Seed for reproducible bootstrap sampling.
- **Default**: None (random seed each run)
- **Type**: Integer
- **Note**: If set, top-N modeling uses `random_state + 10` for different sampling

#### `--top_n TOP_N`
Number of features to retain for second-round modeling.
- **Default**: 600
- **Impact**: Higher values include more features but may reduce specificity

### Data Processing

#### `--normalize_sample_weights`
Normalize bootstrap sample weights to sum to 1.
- **Default**: False
- **Use case**: When sample weights need explicit normalization

#### `--scale_by_std`
Scale the model matrix by standard deviation (without centering).
- **Default**: False
- **Implementation**: Uses `StandardScaler(with_mean=False, with_std=True)`
- **Effect**: Data is scaled but not centered; estimator still fits intercept (`fit_intercept=True`)
- **Use case**: When features have very different scales but you want to preserve the mean structure

### Feature Engineering

#### `--row_max`
Include row maximum as an additional predictor in all-data modeling.
- **Default**: False
- **Description**: Adds the maximum binding value across all TFs for each gene

#### `--squared_pTF`
Include squared perturbed TF term in all-data modeling.
- **Default**: False
- **Mathematical**: Adds `pTF²` term to capture non-linear relationships

#### `--cubic_pTF`
Include cubic perturbed TF term in all-data modeling.
- **Default**: False
- **Mathematical**: Adds `pTF³` term for higher-order non-linearities

#### `--ptf_main_effect`
Include perturbed transcription factor main effect in modeling formula.
- **Default**: False
- **Description**: Adds the pTF binding value as a direct predictor

#### `--exclude_interactor_variables EXCLUDE_INTERACTOR_VARIABLES`
Comma-separated list of variables to exclude from interaction terms.
- **Format**: `var1,var2,var3` or `exclude_all`
- **Example**: `--exclude_interactor_variables "red_median,green_median"`

#### `--add_model_variables ADD_MODEL_VARIABLES`
Comma-separated list of additional variables for all-data modeling.
- **Format**: `var1,var2,var3`
- **Example**: `--add_model_variables "red_median,green_median"`
- **Effect**: Adds `... + red_median + green_median` to model formula

### Binning Options

#### `--bins BINS`
Comma-separated bin edges for data stratification.
- **Default**: `"0,8,12,np.inf"`
- **Format**: Numbers or `np.inf` for infinity
- **Example**: `--bins "0,5,10,15,np.inf"`
- **Purpose**: Stratifies data for cross-validation

### Model Parameters

#### `--all_data_ci_level ALL_DATA_CI_LEVEL`
Confidence interval threshold (%) for selecting significant coefficients in first stage.
- **Default**: 98.0
- **Range**: 80.0-99.9
- **Impact**: Higher values are more stringent for feature selection

#### `--topn_ci_level TOPN_CI_LEVEL`
Confidence interval threshold for second round of modeling.
- **Default**: 90.0
- **Range**: 80.0-99.9
- **Typically**: Lower than `all_data_ci_level` for refinement

#### `--max_iter MAX_ITER`
Maximum iterations for LassoCV convergence.
- **Default**: 10000
- **Range**: 1000-100000
- **Increase if**: Convergence warnings appear

#### `--iterative_dropout`
Enable iterative variable dropout based on confidence intervals.
- **Default**: False
- **Description**: Progressively removes non-significant variables during modeling

#### `--stabilization_ci_start STABILIZATION_CI_START`
Starting confidence interval for iterative dropout stabilization.
- **Default**: 50.0
- **Range**: 50.0-95.0
- **Used with**: `--iterative_dropout`

#### `--stage4_lasso`
Use LassoCV-based interactor significance testing in Stage 4.
- **Default**: False (uses linear regression)
- **Alternative**: More conservative approach with regularization

#### `--stage4_topn`
Perform Stage 4 evaluation on top-n data instead of all data.
- **Default**: False (uses all data)
- **Effect**: Focuses final analysis on high-performing subset

### Output Control

#### `--output_dir OUTPUT_DIR`
Base directory for saving results.
- **Default**: `"./linear_perturbation_binding_modeling_results"`
- **Structure**: Creates subdirectory per run with timestamp

#### `--output_suffix OUTPUT_SUFFIX`
Suffix to append to output subdirectory name.
- **Default**: Empty string
- **Naming**: `{perturbed_tf}{suffix}_{timestamp}`
- **Example**: `YPD1_custom_run_20240115_143022`

### System Options

#### `--n_cpus N_CPUS`
Number of CPU cores for parallel processing.
- **Default**: 4
- **Recommendation**: Match your system's available cores
- **Impact**: Each LassoCV call uses specified cores

## Output Structure

Results are saved in a timestamped subdirectory:

```
{output_dir}/{perturbed_tf}{output_suffix}_{timestamp}/
├── all_data_results/
│   ├── bootstrap_coefficients.csv      # Coefficient estimates per bootstrap
│   ├── confidence_intervals.csv        # CI and significance results
│   ├── model_statistics.csv           # R², CV scores, feature counts
│   ├── feature_importance.csv         # Ranked feature importance
│   └── diagnostic_plots/              # Visualization outputs
├── topn_results/                       # Same structure for top-N analysis
│   ├── bootstrap_coefficients.csv
│   ├── confidence_intervals.csv
│   ├── model_statistics.csv
│   ├── feature_importance.csv
│   └── diagnostic_plots/
├── interactor_significance/
│   ├── significance_results.csv        # Interaction vs main effect tests
│   ├── comparison_statistics.csv       # Statistical comparisons
│   ├── final_selection.csv            # Selected significant interactions
│   └── interaction_plots/             # Interaction visualizations
├── input_data/
│   ├── processed_response.csv         # Cleaned response data
│   ├── processed_predictors.csv       # Cleaned predictor data
│   ├── bootstrap_indices.csv          # Bootstrap sample indices
│   └── data_summary.json             # Data processing summary
└── tfbpmodeling_{timestamp}.log       # Complete execution log
```

## Examples

### Basic Analysis
```bash
python -m tfbpmodeling linear_perturbation_binding_modeling \
    --response_file data/expression.csv \
    --predictors_file data/binding.csv \
    --perturbed_tf YPD1
```

### Reproducible High-Quality Analysis
```bash
python -m tfbpmodeling linear_perturbation_binding_modeling \
    --response_file data/expression.csv \
    --predictors_file data/binding.csv \
    --perturbed_tf YPD1 \
    --n_bootstraps 2000 \
    --top_n 500 \
    --all_data_ci_level 95.0 \
    --topn_ci_level 85.0 \
    --random_state 42 \
    --output_dir ./results \
    --output_suffix _high_quality
```

### Advanced Feature Engineering
```bash
python -m tfbpmodeling linear_perturbation_binding_modeling \
    --response_file data/expression.csv \
    --predictors_file data/binding.csv \
    --perturbed_tf YPD1 \
    --blacklist_file data/exclude_genes.txt \
    --row_max \
    --squared_pTF \
    --cubic_pTF \
    --ptf_main_effect \
    --add_model_variables "red_median,green_median" \
    --exclude_interactor_variables "batch_effect" \
    --normalize_sample_weights \
    --scale_by_std \
    --bins "0,5,10,15,np.inf"
```

### High-Performance Analysis
```bash
python -m tfbpmodeling linear_perturbation_binding_modeling \
    --response_file data/expression.csv \
    --predictors_file data/binding.csv \
    --perturbed_tf YPD1 \
    --n_bootstraps 5000 \
    --n_cpus 16 \
    --max_iter 20000 \
    --iterative_dropout \
    --stage4_lasso \
    --stage4_topn
```

## Performance Considerations

### Memory Usage
- Scales with: features × samples × bootstrap samples
- Large datasets: reduce `n_bootstraps` or `top_n`
- Monitor with: `htop` or `top` during execution

### Runtime Estimation
For typical datasets (1000 features, 100 samples):
- 1000 bootstraps: 10-30 minutes
- 2000 bootstraps: 20-60 minutes
- 5000 bootstraps: 1-3 hours

### Optimization Tips
1. Start with default parameters
2. Use `--random_state` for reproducible development
3. Increase `--n_cpus` to match available cores
4. Use `--log-level DEBUG` for detailed progress tracking
5. Consider `--iterative_dropout` for feature-rich datasets

## Error Handling

### Common Issues

#### File Format Errors
```bash
# Verify CSV format
head -5 data/expression.csv
head -5 data/binding.csv
```

#### Missing Perturbed TF
```bash
# Check column names
head -1 data/expression.csv | tr ',' '\n' | grep YPD1
```

#### Convergence Issues
```bash
# Increase iterations
--max_iter 20000
```

#### Memory Issues
```bash
# Reduce computational load
--n_bootstraps 500 --top_n 300
```

## Related Commands

- **[CLI Overview](overview.md)**: General CLI documentation
- **[Tutorials](../tutorials/basic-workflow.md)**: Step-by-step examples
- **[API Reference](../api/interface.md)**: Programmatic usage