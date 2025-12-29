from mlimputer import MLimputer
from mlimputer.schemas.parameters import imputer_parameters
from mlimputer.data.data_generator import ImputationDatasetGenerator

from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Generate Dataset
# ============================================================================
print("="*60)
print("MLIMPUTER BASIC USAGE")
print("="*60)

# Generate dataset with missing values
generator = ImputationDatasetGenerator(random_state=42)
X, y = generator.quick_multiclass(n_samples=2000, missing_rate=0.15, n_categorical=5)

# Split your data and reset index
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
X_train = X_train.reset_index(drop=True)  # Reset index - Required
X_test = X_test.reset_index(drop=True)    # Reset index - Required

# Configure imputer parameters
params = imputer_parameters()
params["KNN"]["n_neighbors"] = 5
params["KNN"]["weights"] = "distance"

# Create imputer
imputer = MLimputer(
    imput_model="KNN",
    imputer_configs=params
)

# Fit on training data
imputer.fit(X_train)

# Transform both sets
X_train_imputed = imputer.transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Check results
print("\n" + "="*60)
print("IMPUTATION RESULTS")
print("="*60)

print("\nTraining Set:")
print(f"  Missing values: {X_train.isnull().sum().sum():,} → {X_train_imputed.isnull().sum().sum():,}")
print(f"  Imputed: {X_train.isnull().sum().sum():,} values")

print("\nTest Set:")
print(f"  Missing values: {X_test.isnull().sum().sum():,} → {X_test_imputed.isnull().sum().sum():,}")
print(f"  Imputed: {X_test.isnull().sum().sum():,} values")




