import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

from mlimputer import MLimputer
from mlimputer.evaluation.evaluator import Evaluator
from mlimputer.schemas.parameters import imputer_parameters, update_model_config
from mlimputer.data.data_generator import ImputationDatasetGenerator
from mlimputer.utils.serialization import ModelSerializer

import warnings
warnings.filterwarnings("ignore")

print("="*60)
print("MLIMPUTER - PERFORMANCE EVALUATION")
print("="*60)

# ============================================================================
# Generate Dataset
# ============================================================================
generator = ImputationDatasetGenerator(random_state=42)

TASK = "binary_classification"
X, y = generator.quick_binary(n_samples=2000, missing_rate=0.15)

predictive_models = [
    RandomForestClassifier(n_estimators=50, random_state=42),
    ExtraTreesClassifier(n_estimators=50, random_state=42),
    GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, random_state=42),
    DecisionTreeClassifier(max_depth=10, random_state=42),
]

primary_metric = "F1"

print(f"\nTask: {TASK}")
print(f"Dataset: {X.shape}")
print(f"Missing: {X.isnull().sum().sum()} values")
print(f"Models: {[m.__class__.__name__ for m in predictive_models]}")

# Split data
data = pd.concat([X, y], axis=1)
train_size = int(0.8 * len(data))
train = data.iloc[:train_size].reset_index(drop=True)
test = data.iloc[train_size:].reset_index(drop=True)

# ============================================================================
# Configure Imputation Strategies
# ============================================================================
print("\n" + "="*60)
print("IMPUTATION CONFIGURATION")
print("="*60)

params = imputer_parameters()

# Optimize parameters
params["RandomForest"] = update_model_config(
    "RandomForest",
    {"n_estimators": 50, "max_depth": 10}
)
params["ExtraTrees"]["n_estimators"] = 50
params["GBR"]["learning_rate"] = 0.05
params["KNN"]["n_neighbors"] = 7

strategies = ["RandomForest", "ExtraTrees", "GBR", "KNN"]
print(f"Strategies: {strategies}")

# ============================================================================
# Cross-Validation Evaluation
# ============================================================================
print("\n" + "="*60)
print("CROSS-VALIDATION (3-fold)")
print("="*60)

evaluator = Evaluator(
    imputation_models=strategies,
    train=train,
    target="target",
    n_splits=3,
    hparameters=params,
    problem_type=TASK
)

# Run evaluation
cv_results = evaluator.evaluate_imputation_models(models=predictive_models)

# Get best strategy
best_imputer = evaluator.get_best_imputer()
print(f"\n✓ Best imputation strategy: {best_imputer}")

# Show top results
aggregate = cv_results[cv_results["Fold"] == "Aggregate"]
metric_col = f"{primary_metric} Mean"
top_results = aggregate.nlargest(5, metric_col)

print(f"\nTop strategies by {primary_metric}:")
print(top_results[["Model", "Imputer Model", metric_col]].to_string(index=False))

# ============================================================================
# Test Set Evaluation
# ============================================================================
print("\n" + "="*60)
print("TEST SET EVALUATION")
print("="*60)

test_results = evaluator.evaluate_test_set(
    test=test,
    imput_model=best_imputer,
    models=predictive_models
)

print("\nTest Performance:")
print(test_results.to_string(index=False))

# ============================================================================
# Save Results
# ============================================================================
print("\n" + "="*60)
print("EXPORT")
print("="*60)

# Fit best imputer on full train set
best_imputer_model = MLimputer(
    imput_model=best_imputer,
    imputer_configs=params
)
best_imputer_model.fit(X=train.drop(columns=['target']))

# Save configuration
best_config = {
    "strategy": best_imputer,
    "parameters": params.get(best_imputer, {}),
    "task": TASK,
    "primary_metric": primary_metric
}

ModelSerializer.save(
    obj=best_imputer_model,
    filepath="best_imputer.joblib",
    format="joblib",
    metadata=best_config
)

print("✓ Config saved: best_config.json")
print("✓ Model saved: best_imputer.joblib")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*60)
print("EVALUATION SUMMARY")
print("="*60)

summary = evaluator.get_summary_report()
print(f"\nDataset: {summary['dataset_shape'][0]} samples, {summary['dataset_shape'][1]} features")
print(f"Task: {TASK}")
print(f"Best imputer: {best_imputer}")
print(f"Primary metric: {primary_metric}")
print(f"Strategies tested: {len(strategies)}")
print(f"Models evaluated: {len(predictive_models)}")

print("\n" + "="*60)
print("EVALUATION COMPLETED")
print("="*60)