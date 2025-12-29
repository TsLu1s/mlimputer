import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression #, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso

from mlimputer import MLimputer
from mlimputer.evaluation.evaluator import Evaluator
from mlimputer.evaluation.cross_validation import CrossValidator, CrossValidationConfig
from mlimputer.schemas.parameters import imputer_parameters, update_model_config
from mlimputer.data.data_generator import ImputationDatasetGenerator
from mlimputer.utils.splitter import DataSplitter
from mlimputer.utils.serialization import ModelSerializer

import warnings
warnings.filterwarnings("ignore")

print("="*60)
print("MLIMPUTER EVALUATION")
print("="*60)

# ============================================================================
# Generate Dataset
# ============================================================================
generator = ImputationDatasetGenerator(random_state=42)

# Choose task type
TASK = "multiclass_classification"  # Options: "regression", "binary_classification", "multiclass_classification"
if TASK == "regression":
    X, y = generator.quick_regression(n_samples=3000, missing_rate=0.15)
    predictive_models = [
        LinearRegression(),
        RandomForestRegressor(n_estimators=50, random_state=42),
        ExtraTreesRegressor(n_estimators=50, random_state=42),
        GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, random_state=42),
        Ridge(alpha=1.0),
        Lasso(alpha=1.0),
    ]
    primary_metric = "MAE"  # Options: "MAE", "RMSE", "R2", "MAPE"
    stratify = None
    
elif TASK == "binary_classification":
    X, y = generator.quick_binary(n_samples=3000, missing_rate=0.15)
    predictive_models = [                                                                        
        RandomForestClassifier(n_estimators=50, random_state=42),
        ExtraTreesClassifier(n_estimators=50, random_state=42),
        GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, random_state=42),
        DecisionTreeClassifier(max_depth=10, random_state=42),
        AdaBoostClassifier(n_estimators=50, random_state=42)
    ]
    primary_metric = "F1"  # Options: "Accuracy", "Precision", "Recall", "F1"
    stratify = y

else:  # multiclass_classification
    X, y = generator.quick_multiclass(n_samples=3000, n_classes=3, missing_rate=0.15)
    predictive_models = [                                                                        
        RandomForestClassifier(n_estimators=50, random_state=42),
        #LogisticRegression(max_iter=1000, random_state=42),
        ExtraTreesClassifier(n_estimators=50, random_state=42),
        GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, random_state=42),
        DecisionTreeClassifier(max_depth=10, random_state=42),
        AdaBoostClassifier(n_estimators=50, random_state=42)
    ]
    primary_metric = "F1"  # Options: "Accuracy", "Precision", "Recall", "F1"
    stratify = y
    

print(f"\nTask: {TASK}")
print(f"Dataset: {X.shape}")
print(f"Missing: {X.isnull().sum().sum()} values")
print(f"Models: {[m.__class__.__name__ for m in predictive_models]}")

# Split using DataSplitter for automatic index reset
splitter = DataSplitter(random_state=42)
data = pd.concat([X, y], axis=1)

# Create train/test split for evaluation  
train_size = int(0.8 * len(data))
train = data.iloc[:train_size].reset_index(drop=True)
test = data.iloc[train_size:].reset_index(drop=True)

test.dtypes

# ============================================================================
# Configure Imputation
# ============================================================================
print("\n" + "="*60)
print("IMPUTATION CONFIGURATION")
print("="*60)

params = imputer_parameters()

# Optimize parameters for each strategy
params["RandomForest"] = update_model_config(
    "RandomForest",
    {"n_estimators": 50, "max_depth": 10}
)
params["ExtraTrees"]["n_estimators"] = 50
params["GBR"]["learning_rate"] = 0.05
params["KNN"]["n_neighbors"] = 7

strategies = ["RandomForest", "ExtraTrees", "GBR", "KNN"]  # All: + "XGBoost", "Catboost"
print(f"Strategies: {strategies}")

# ============================================================================
# Standard Evaluation
# ============================================================================
print("\n" + "="*60)
print("STANDARD CROSS-VALIDATION (3-fold)")
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
print(f"\nBest strategy: {best_imputer}")

# Show top results
aggregate = cv_results[cv_results["Fold"] == "Aggregate"]  # Options: "Fold", "Model", "Imputer Model"
metric_col = f"{primary_metric} Mean"                      # Options: "Mean", "Std"

if TASK == "regression":
    top_results = aggregate.nsmallest(5, metric_col)
else:
    top_results = aggregate.nlargest(5, metric_col)

## Remove Duplicated Columns
print(f"\nTop strategies by {primary_metric}:")
print(top_results[["Model", "Imputer Model", metric_col]].to_string(index=False))

# ============================================================================
# Custom Cross-Validation
# ============================================================================
print("\n" + "="*60)
print("CUSTOM CROSS-VALIDATION (5-fold)")
print("="*60)

# Create configuration
custom_config = CrossValidationConfig(
    n_splits=5,
    shuffle=True,
    random_state=123,
    verbose=1
)

custom_validator = CrossValidator(config=custom_config)

# Prepare imputed data using best strategy
best_imputer_model = MLimputer(
    imput_model=best_imputer,
    imputer_configs=params
)

# FIT WITHOUT TARGET
best_imputer_model.fit(X=train.drop(columns=['target']))

# TRANSFORM WITHOUT TARGET
X_train_imputed = best_imputer_model.transform(X=train.drop(columns=['target']))

# ADD TARGET BACK AFTER IMPUTATION
X_train_imputed['target'] = train['target'].values

# Run custom validation
print(f"\nRunning custom {custom_config.n_splits}-fold CV...")
custom_results = custom_validator.validate(
    X=X_train_imputed,
    target='target',
    models=predictive_models,
    problem_type=TASK
)

# Get leaderboard
leaderboard = custom_validator.get_leaderboard()
print("\nCustom CV Leaderboard:")
print(leaderboard.head(5))

# ============================================================================
# Report Analysis
# ============================================================================
print("\n" + "="*60)
print("REPORT ANALYSIS")
print("="*60)

imputation_report = evaluator.get_summary_report()

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
print(test_results)

# ============================================================================
# Per-Strategy Evaluation
# ============================================================================

print("\n" + "="*60)
print("PER-STRATEGY EVALUATION")
print("="*60)

strategy_results = {}
for strategy in strategies[:2]:
    # Create and fit imputer
    strategy_imputer = MLimputer(imput_model=strategy, imputer_configs=params)
    strategy_imputer.fit(X=train.drop(columns=['target']).copy())
    X_imputed = strategy_imputer.transform(X=train.copy())
    
    # Run validation
    strategy_cv = custom_validator.validate(
        X=X_imputed,
        target="target",                                        # Just a name reference
        y=X_imputed['target'],                                  # Actual target values
        models=[predictive_models[0]],
        problem_type=TASK
    )
    
    strategy_results[strategy] = strategy_cv.fold_results[0].metrics[primary_metric.lower()]

print(f"\nPer-strategy scores ({primary_metric}):")
for strategy, score in strategy_results.items():
    print(f"  {strategy}: {score:.4f}")

strategy_results[strategy] = strategy_cv.fold_results[0].metrics

print(strategy_results)

# ============================================================================
# Save Results
# ============================================================================
print("\n" + "="*60)
print("EXPORT")
print("="*60)

# Save configuration
best_config = {
    "strategy": best_imputer,
    "parameters": params.get(best_imputer, {}),
    "task": TASK,
    "primary_metric": primary_metric
}

# Save fitted imputer
ModelSerializer.save(
    obj=best_imputer_model,
    filepath="best_imputer.joblib",
    format="joblib",
    metadata=best_config
)

print(" Config saved: best_config.json")
print(" Model saved: best_imputer.joblib")

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
print("Evaluation completed!")
print("="*60)








