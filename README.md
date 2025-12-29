[![LinkedIn][linkedin-shield]][linkedin-url]
[![Contributors][contributors-shield]][contributors-url]
[![Stargazers][stars-shield]][stars-url]
[![MIT License][license-shield]][license-url]
[![Downloads][downloads-shield]][downloads-url]
[![Month Downloads][downloads-month-shield]][downloads-month-url]

[contributors-shield]: https://img.shields.io/github/contributors/TsLu1s/MLimputer.svg?style=for-the-badge&logo=github&logoColor=white
[contributors-url]: https://github.com/TsLu1s/MLimputer/graphs/contributors
[stars-shield]: https://img.shields.io/github/stars/TsLu1s/MLimputer.svg?style=for-the-badge&logo=github&logoColor=white
[stars-url]: https://github.com/TsLu1s/MLimputer/stargazers
[license-shield]: https://img.shields.io/github/license/TsLu1s/MLimputer.svg?style=for-the-badge&logo=opensource&logoColor=white
[license-url]: https://github.com/TsLu1s/MLimputer/blob/main/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/luisfssantos98/
[downloads-shield]: https://static.pepy.tech/personalized-badge/mlimputer?period=total&units=international_system&left_color=grey&right_color=blue&left_text=Total%20Downloads
[downloads-url]: https://pepy.tech/project/mlimputer
[downloads-month-shield]: https://static.pepy.tech/personalized-badge/mlimputer?period=month&units=international_system&left_color=grey&right_color=blue&left_text=Month%20Downloads
[downloads-month-url]: https://pepy.tech/project/mlimputer

<br>
<p align="center">
  <h2 align="center"> MLimputer: Missing Data Imputation Framework for Machine Learning
  <br>
  
## Framework Contextualization: Advanced Missing Data Imputation for Tabular Data

The `MLimputer` project provides a comprehensive and integrated framework to automate the handling of missing values in datasets through advanced machine learning imputation. It aims to reduce bias and increase the precision of imputation results compared to traditional methods by leveraging supervised learning algorithms.

This package offers multiple algorithm options to impute your data, where each column with missing values is predicted using robust preprocessing and state-of-the-art machine learning models.

The architecture includes three main components:
* **Missing Data Analysis**: Automatic detection and pattern analysis of missing values
* **Data Preprocessing**: Intelligent handling of categorical and numerical features
* **Supervised Model Imputation**: Multiple ML algorithms for accurate value prediction

### Key Capabilities

* **General applicability on tabular datasets**: Works with any tabular data for both regression and classification tasks
* **Robust preprocessing**: Automatic handling of categorical encoding and feature scaling
* **Multiple imputation strategies**: Choose from 7 different ML algorithms based on your data characteristics
* **Performance evaluation**: Built-in evaluation framework to compare and select the best imputation strategy
* **Production ready**: Save and load fitted imputers for deployment

### Main Development Tools

Major frameworks used to build this project:
* [Pandas](https://pandas.pydata.org/) - Data manipulation and analysis
* [Scikit-learn](https://scikit-learn.org/stable/) - Core ML algorithms
* [XGBoost](https://xgboost.ai/) - Gradient boosting
* [CatBoost](https://catboost.ai/) - Gradient boosting with categorical support
* [Pydantic](https://pydantic-docs.helpmanual.io/) - Data validation

## Installation

Binary installer for the latest released version is available at the Python Package Index [(PyPI)](https://pypi.org/project/mlimputer/).

```bash
pip install mlimputer
```

GitHub Project Link: [https://github.com/TsLu1s/MLimputer](https://github.com/TsLu1s/MLimputer)

## Quick Start Guide

### Basic Usage Example

The first step is to import the package, load your dataset, and choose an imputation model. Available imputation models are:
* `RandomForest` 
* `ExtraTrees` 
* `GBR` 
* `KNN` 
* `XGBoost` 
* `Catboost` 

```python
import pandas as pd
from mlimputer import MLimputer
from mlimputer.schemas.parameters import imputer_parameters
from mlimputer.utils.splitter import DataSplitter
import warnings
warnings.filterwarnings("ignore")

# Load your data
data = pd.read_csv('your_dataset.csv')

# Split with automatic index reset (required for MLimputer)
splitter = DataSplitter(random_state=42)
X_train, X_test, y_train, y_test = splitter.split(
    data.drop(columns=['target']), 
    data['target'], 
    test_size=0.2
)

# Configure imputation parameters (optional)
params = imputer_parameters()
params["RandomForest"]["n_estimators"] = 50
params["RandomForest"]["max_depth"] = 10

# Create and fit imputer
imputer = MLimputer(imput_model="RandomForest", imputer_configs=params)
imputer.fit(X=X_train)

# Transform datasets
X_train_imputed = imputer.transform(X=X_train)
X_test_imputed = imputer.transform(X=X_test)

# Save fitted imputer for production use
import pickle
with open("fitted_imputer.pkl", 'wb') as f:
    pickle.dump(imputer, f)
```

### Advanced Configuration

Customize imputation model hyperparameters for better performance:

```python
from mlimputer.schemas.parameters import imputer_parameters, update_model_config

# Get default parameters
params = imputer_parameters()

# Method 1: Direct modification
params["KNN"]["n_neighbors"] = 7
params["KNN"]["weights"] = "distance"

# Method 2: Using update function with validation
params["RandomForest"] = update_model_config(
    "RandomForest",
    {"n_estimators": 100, "max_depth": 15, "min_samples_split": 5}
)

# Apply different strategies
strategies = ["RandomForest", "KNN", "XGBoost"]
for strategy in strategies:
    imputer = MLimputer(imput_model=strategy, imputer_configs=params)
    imputer.fit(X=X_train)
    print(f"{strategy}: {imputer.get_summary()['n_columns_imputed']} columns imputed")
```

## Performance Evaluation

The MLimputer framework includes a robust evaluation module to assess and compare different imputation strategies. This helps you select the most effective approach for your specific dataset.

### Evaluation Framework

```python
from mlimputer.evaluation.evaluator import Evaluator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression

# Define evaluation parameters
imputation_strategies = ["RandomForest", "ExtraTrees", "GBR", "KNN"]

# Choose models based on your task
if target.dtype == "object":  # Classification
    models = [
        LogisticRegression(max_iter=1000),
        RandomForestClassifier(n_estimators=50)
    ]
else:  # Regression
    models = [
        LinearRegression(),
        RandomForestRegressor(n_estimators=50)
    ]

# Initialize evaluator
evaluator = Evaluator(
    imputation_models=imputation_strategies,
    train=train_data,
    target="target_column",
    n_splits=3,  # Cross-validation folds
    hparameters=params
)

# Run cross-validation evaluation
cv_results = evaluator.evaluate_imputation_models(models=models)

# Get best performing imputation strategy
best_imputer = evaluator.get_best_imputer()
print(f"Best imputation strategy: {best_imputer}")

# Evaluate on test set
test_results = evaluator.evaluate_test_set(
    test=test_data,
    imput_model=best_imputer,
    models=models
)
```

### Custom Cross-Validation

For more control over the evaluation process:

```python
from mlimputer.evaluation.cross_validation import CrossValidator, CrossValidationConfig

# Configure custom cross-validation
custom_config = CrossValidationConfig(
    n_splits=5,
    shuffle=True,
    random_state=42,
    verbose=1
)

# Create validator
validator = CrossValidator(config=custom_config)

# Run validation
results = validator.validate(
    X=X_imputed,
    target='target',
    y=y,
    models=models,
    problem_type="regression"  # or "binary_classification", "multiclass_classification"
)

# Get leaderboard
leaderboard = validator.get_leaderboard()
print(leaderboard.head())
```

## Working with Generated Data

MLimputer includes utilities for generating datasets with missing values for testing:

```python
from mlimputer.data.dataset_generator import ImputationDatasetGenerator

generator = ImputationDatasetGenerator(random_state=42)

# Regression dataset
X_reg, y_reg = generator.quick_regression(
    n_samples=2000, 
    missing_rate=0.15
)

# Binary classification
X_bin, y_bin = generator.quick_binary(
    n_samples=2000, 
    missing_rate=0.15
)

# Multiclass classification
X_multi, y_multi = generator.quick_multiclass(
    n_samples=2000,
    n_classes=4,
    missing_rate=0.15,
    n_categorical=3  # Include categorical features
)
```

## Production Deployment

### Saving and Loading Models
```python
from mlimputer.utils.serialization import ModelSerializer

# Save with metadata
ModelSerializer.save(
    obj=imputer,
    filepath="production_imputer.joblib",
    format="joblib",
    metadata={
        "model": "RandomForest",
        "train_shape": X_train.shape,
        "version": "1.0"
    }
)

# Load with metadata
loaded_imputer, metadata = ModelSerializer.load_with_metadata(
    filepath="production_imputer.joblib",
    format="joblib"
)

# Use loaded imputer on new data
new_data_imputed = loaded_imputer.transform(new_data)
```

## Important Notes

* **Index Reset Required**: Always use `DataSplitter` or reset indices manually after splitting data
* **Categorical Handling**: The framework automatically detects and encodes categorical columns
* **Missing Pattern Preservation**: The imputer learns missing patterns from training data for consistent imputation
* **Memory Efficient**: Large datasets are processed in batches automatically

## Example Notebooks

### 1. Basic Usage Example
A complete walkthrough demonstrating fundamental imputation workflow:
- Dataset generation with controlled missing patterns
- Train/test splitting with automatic index handling  
- Model configuration and fitting
- Imputation and evaluation
- Saving fitted models for production

[View Basic Example](https://github.com/TsLu1s/mlimputer/blob/main/examples/basic_usage.py) 

### 2. Performance Evaluation Example
Comprehensive evaluation comparing multiple imputation strategies:
- Cross-validation setup for robust evaluation
- Comparison of 7 different imputation algorithms
- Custom evaluation configurations
- Best model selection based on metrics
- Production deployment preparation

[View Evaluation Example](https://github.com/TsLu1s/mlimputer/blob/main/examples/performance_evaluation.py)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use MLimputer in your research, please cite:
```bibtex
@software{mlimputer,
  author = {Luis Fernando Santos},
  title = {MLimputer: Missing Data Imputation Framework for Supervised Machine Learning},
  year = {2023},
  url = {https://github.com/TsLu1s/MLimputer}
}
```
    
## License

Distributed under the MIT License. See [LICENSE](https://github.com/TsLu1s/TSForecasting/blob/main/LICENSE) for more information.

## Contact 
 
Luis Santos - [LinkedIn](https://www.linkedin.com/in/luisfssantos98/)

