import pandas as pd
from sklearn.model_selection import train_test_split
from featransform.utils.data_generator import DatasetGenerator
from featransform.processing.imputation import CoreImputer
from featransform.processing.encoding import CategoricalEncoder
from featransform.features.anomaly import AnomalyEnsemble, IsolationForestStrategy, LocalOutlierFactorStrategy, OneClassSVMStrategy, EllipticEnvelopeStrategy
from featransform.features.clustering import ClusteringEnsemble, KMeansStrategy, BirchStrategy, MiniBatchKMeansStrategy, GaussianMixtureStrategy,  DBSCANStrategy
from featransform.features.dimensionality import DimensionalityEnsemble, PCAStrategy
from featransform.optimization.selector import FeatureSelector
from featransform.core.enums import (
    ImputationStrategy, EncodingStrategy, SelectionStrategy
)

print("\n" + "=" * 70)
print(" COMPLETE PIPELINE - TRAIN/TEST FLOW")
print("=" * 70)

# Generate complex dataset
X_full, y_full = DatasetGenerator.complex_dataset(
    n_samples=10000,
    task='binary_classification'  # Options: 'regression', 'binary_classification', 'multiclass_classification'
)

# Split train/test
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X_full, y_full, test_size=0.3, random_state=42
)
print("\nComplex dataset shapes:")
print(f"  Train: {X_train_full.shape}")
print(f"  Test: {X_test_full.shape}")
print(f"  Dtypes: {X_full.dtypes.value_counts().to_dict()}")
print(f"  Train missing: {X_train_full.isnull().sum().sum()}")
print(f"  Test missing: {X_test_full.isnull().sum().sum()}")

# Step 2: Encoding
print("\n2. Categorical encoding...")
# Available encoding strategies:
# - EncodingStrategy.LABEL: Label encoding for ordinal categories
encoder = CategoricalEncoder(strategy=EncodingStrategy.LABEL)
encoder.fit(X_train_full, y_train_full)
X_train_proc = encoder.transform(X_train_full)
X_test_proc = encoder.transform(X_test_full)
print(f"  Shape after encoding: Train {X_train_proc.shape}, Test {X_test_proc.shape}")

# Step 3: Imputation
print("\n3. Imputation...")
# Available imputation strategies:
# - ImputationStrategy.MEAN: Replace with mean (numeric only)
# - ImputationStrategy.MEDIAN: Replace with median (numeric only)
# - ImputationStrategy.MODE: Replace with most frequent value
# - ImputationStrategy.ITERATIVE: Iterative imputation using other features
# - ImputationStrategy.KNN: K-Nearest Neighbors imputation
imputer = CoreImputer(strategy=ImputationStrategy.ITERATIVE)
imputer.fit(X_train_proc)
X_train_proc = imputer.transform(X_train_proc)
X_test_proc = imputer.transform(X_test_proc)
print(f"  Train missing after: {X_train_proc.isnull().sum().sum()}")
print(f"  Test missing after: {X_test_proc.isnull().sum().sum()}")

# Step 4: Feature engineering
print("\n4. Feature engineering...")

# ANOMALY DETECTION OPTIONS:
# Available strategies (can use multiple):
# - IsolationForestStrategy(n_estimators=100, contamination=0.1, include_scores=False)
# - LocalOutlierFactorStrategy(n_neighbors=20, contamination=0.1, include_scores=False)
# - OneClassSVMStrategy(nu=0.05, kernel='rbf', contamination=0.1, include_scores=False)
# - EllipticEnvelopeStrategy(contamination=0.1, include_scores=False)
anomaly_strategies = [
    IsolationForestStrategy(n_estimators=100, contamination=0.1),
    # LocalOutlierFactorStrategy(n_neighbors=20),
    # OneClassSVMStrategy(nu=0.05, kernel='rbf'),
    # EllipticEnvelopeStrategy(contamination=0.1)  # May cause warnings with high-dim data
]
anomaly_ens = AnomalyEnsemble(strategies=anomaly_strategies, verbose=False)
anomaly_ens.fit(X_train_proc)
train_anom = anomaly_ens.transform(X_train_proc).features
test_anom = anomaly_ens.transform(X_test_proc).features

# CLUSTERING OPTIONS:
# Available strategies (can use multiple):
# - KMeansStrategy(n_clusters=3, random_state=42)
# - BirchStrategy(n_clusters=3, threshold=0.5, branching_factor=50)
# - MiniBatchKMeansStrategy(n_clusters=3, batch_size=100, random_state=42)
# - DBSCANStrategy(eps=0.5, min_samples=5)  # Note: doesn't use n_clusters
# - GaussianMixtureStrategy(n_clusters=3, covariance_type='full', random_state=42)
#   covariance_type options: 'full', 'tied', 'diag', 'spherical'
clustering_strategies = [
    KMeansStrategy(n_clusters=3, random_state=42),
    # BirchStrategy(n_clusters=3, threshold=0.5),
    # MiniBatchKMeansStrategy(n_clusters=3, batch_size=100),
    # DBSCANStrategy(eps=0.5, min_samples=5),
    # GaussianMixtureStrategy(n_clusters=3, covariance_type='full')
]
cluster_ens = ClusteringEnsemble(strategies=clustering_strategies, verbose=False)
cluster_ens.fit(X_train_proc)
train_clust = cluster_ens.transform(X_train_proc).features
test_clust = cluster_ens.transform(X_test_proc).features

# DIMENSIONALITY REDUCTION
# Available strategies:
# - PCAStrategy(n_components=0.95)  # Keep 95% variance or specify number
# - TruncatedSVDStrategy(n_components=10)  # For sparse data
# - FastICAStrategy(n_components=10, algorithm='parallel', whiten='unit-variance')
print("  - Dimensionality reduction...")
dim_strategies = [
    PCAStrategy(n_components=5),  # Reduce to 5 principal components
    # TruncatedSVDStrategy(n_components=10)
]
dim_ens = DimensionalityEnsemble(strategies=dim_strategies, verbose=False)
dim_ens.fit(X_train_proc, y_train_full)  # Some dim reduction methods benefit from y
train_dim = dim_ens.transform(X_train_proc).features
test_dim = dim_ens.transform(X_test_proc).features
print(f"    Added {train_dim.shape[1]} dimensionality features")

# Combine all features
# Combine all features
X_train_all = pd.concat([
    X_train_proc,    # Original processed features
    train_anom,      # Anomaly detection features
    train_clust,     # Clustering features
    train_dim        # Dimensionality reduction features
], axis=1)

X_test_all = pd.concat([
    X_test_proc,     # Original processed features
    test_anom,       # Anomaly detection features
    test_clust,      # Clustering features
    test_dim         # Dimensionality reduction features
], axis=1)
print("\n  Combined feature matrix:")
print(f"    Train: {X_train_proc.shape[1]} original + {train_anom.shape[1]} anomaly + {train_clust.shape[1]} cluster + {train_dim.shape[1]} dim = {X_train_all.shape[1]} total")
print(f"    Test:  {X_test_proc.shape[1]} original + {test_anom.shape[1]} anomaly + {test_clust.shape[1]} cluster + {test_dim.shape[1]} dim = {X_test_all.shape[1]} total")


# Step 5: Feature selection
print("\n5. Feature selection...")
selector_final = FeatureSelector(
    strategy=SelectionStrategy.IMPORTANCE,
    min_features=10,
    verbose=False
)

# Fit to calculate importance
selector_final.fit(X_train_all, y_train_full)

# Select top 70% features or min_features (whichever is larger)
threshold = 0.7  # Use top 50% of cumulative importance
selected_features = selector_final.select_by_threshold(threshold)
selector_final.set_selected_features(selected_features)

# Transform data
X_train_final = selector_final.transform(X_train_all)
X_test_final = selector_final.transform(X_test_all)
print(f"  Final shapes after selection: Train {X_train_final.shape}, Test {X_test_final.shape}")

# Show which features were selected
print(f"  Selected features: {len(selected_features)} features retained")

# Get importance scores
scores = selector_final.get_feature_scores()
if scores:
    top_10 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]
    print("Top 10 features by importance:")
    for feat, score in top_10:
        print(f"    {feat}: {score:.4f}")

print("\n" + "=" * 70)
print("TRAIN/TEST COMPONENT TESTING COMPLETED SUCCESSFULLY!")
print("=" * 70)