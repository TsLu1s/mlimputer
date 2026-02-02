import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from mlimputer import MLimputer
from mlimputer.evaluation.cross_validation import CrossValidator, CrossValidationConfig
from mlimputer.schemas.parameters import imputer_parameters
from mlimputer.data.data_generator import ImputationDatasetGenerator

import warnings
warnings.filterwarnings("ignore")

print("="*60)
print("MLIMPUTER - SINGULAR ANALYSIS")
print("="*60)

# ============================================================================
# Generate Dataset
# ============================================================================
generator = ImputationDatasetGenerator(random_state=42)
X, y = generator.quick_binary(n_samples=1500, missing_rate=0.20)

print(f"\nDataset: {X.shape}")
print(f"Missing: {X.isnull().sum().sum()} values ({X.isnull().sum().sum()/X.size:.1%})")
print(f"Target distribution:\n{y.value_counts()}")

# Split data
data = pd.concat([X, y], axis=1)
train_size = int(0.8 * len(data))
train = data.iloc[:train_size].reset_index(drop=True)
test = data.iloc[train_size:].reset_index(drop=True)

# ============================================================================
# Configure Single Strategy
# ============================================================================
print("\n" + "="*60)
print("STRATEGY: RANDOM FOREST")
print("="*60)

params = imputer_parameters()
params["RandomForest"]["n_estimators"] = 100
params["RandomForest"]["max_depth"] = 15
params["RandomForest"]["min_samples_split"] = 5

print(f"Configuration:")
for key, value in params["RandomForest"].items():
    print(f"  {key}: {value}")

# ============================================================================
# Fit and Transform
# ============================================================================
print("\n" + "="*60)
print("IMPUTATION")
print("="*60)

imputer = MLimputer(
    imput_model="RandomForest",
    imputer_configs=params
)

# Fit on train (excluding target)
imputer.fit(X=train.drop(columns=['target']))

# Transform both sets
X_train_imputed = imputer.transform(X=train.drop(columns=['target']))
X_test_imputed = imputer.transform(X=test.drop(columns=['target']))

print(f"\n✓ Training imputed: {train.drop(columns=['target']).isnull().sum().sum()} → 0 missing")
print(f"✓ Test imputed: {test.drop(columns=['target']).isnull().sum().sum()} → 0 missing")

# ============================================================================
# Column-wise Analysis
# ============================================================================
print("\n" + "="*60)
print("COLUMN-WISE MISSING DATA")
print("="*60)

train_missing = train.drop(columns=['target']).isnull().sum()
train_missing = train_missing[train_missing > 0].sort_values(ascending=False)

print(f"\nTop 5 columns with missing values (training):")
for col, count in train_missing.head(5).items():
    pct = (count / len(train)) * 100
    print(f"  {col}: {count} ({pct:.1f}%)")

# ============================================================================
# Cross-Validation Analysis
# ============================================================================
print("\n" + "="*60)
print("CROSS-VALIDATION (5-fold)")
print("="*60)

# Add target back for CV
X_train_imputed['target'] = train['target'].values

cv_config = CrossValidationConfig(
    n_splits=5,
    shuffle=True,
    random_state=42,
    verbose=1
)

validator = CrossValidator(config=cv_config)

model = RandomForestClassifier(n_estimators=50, random_state=42)

cv_results = validator.validate(
    X=X_train_imputed,
    target='target',
    models=[model],
    problem_type='binary_classification'
)

leaderboard = validator.get_leaderboard()

# Show fold-by-fold results
print("\nFold-by-Fold Results:")
fold_results = leaderboard[leaderboard['Fold'] != 'Aggregate']
print(fold_results[['Fold', 'f1', 'accuracy', 'precision', 'recall']].to_string(index=False))

# Show aggregate
print("\nAggregate Results:")
agg_results = leaderboard[leaderboard['Fold'] == 'Aggregate']
print(agg_results[['F1 Mean', 'ACCURACY Mean', 'PRECISION Mean', 'RECALL Mean']].to_string(index=False))

# ============================================================================
# Test Set Prediction
# ============================================================================
print("\n" + "="*60)
print("TEST SET PREDICTION")
print("="*60)

# Train final model
final_model = RandomForestClassifier(n_estimators=50, random_state=42)
final_model.fit(X_train_imputed.drop(columns=['target']), train['target'])

# Predict on test
y_pred = final_model.predict(X_test_imputed)
y_true = test['target']

# Classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred))

# Confusion matrix
print("Confusion Matrix:")
cm = confusion_matrix(y_true, y_pred)
print(cm)

# ============================================================================
# Feature Importance
# ============================================================================
print("\n" + "="*60)
print("FEATURE IMPORTANCE")
print("="*60)

importances = final_model.feature_importances_
feature_names = X_train_imputed.drop(columns=['target']).columns
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance_df.head(10).to_string(index=False))

# ============================================================================
# Imputation Summary
# ============================================================================
print("\n" + "="*60)
print("IMPUTATION SUMMARY")
print("="*60)

summary = imputer.get_summary()
print(f"\nStrategy: {summary['model']}")
print(f"Columns imputed: {summary['n_columns_imputed']}")
print(f"Fit timestamp: {summary['fit_timestamp']}")
print(f"Status: {summary['status']}")

print("\n" + "="*60)
print("ANALYSIS COMPLETED")
print("="*60)