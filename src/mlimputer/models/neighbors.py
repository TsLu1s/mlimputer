from sklearn.neighbors import KNeighborsRegressor
from mlimputer.core.base import BaseImputationModel

class KNNImputation(BaseImputationModel):
    """K-Nearest Neighbors imputation model."""
    
    def __init__(
        self,
        n_neighbors: int = 5,
        weights: str = 'uniform',
        algorithm: str = 'auto',
        leaf_size: int = 30,
        p: int = 2,
        **kwargs
    ):
        super().__init__(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            **kwargs
        )
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize KNN model."""
        self.model = KNeighborsRegressor(**self.model_params)
