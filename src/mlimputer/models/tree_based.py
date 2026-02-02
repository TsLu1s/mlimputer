from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from mlimputer.core.base import BaseImputationModel

from typing import Optional


class RandomForestImputation(BaseImputationModel):
    """Random Forest imputation model."""

    def __init__(
        self,
        n_estimators: int = 100,
        random_state: int = 42,
        criterion: str = "squared_error",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str = "sqrt",
        **kwargs,
    ):
        super().__init__(
            n_estimators=n_estimators,
            random_state=random_state,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            **kwargs,
        )
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize Random Forest model."""
        self.model = RandomForestRegressor(**self.model_params)


class ExtraTreesImputation(BaseImputationModel):
    """Extra Trees imputation model."""

    def __init__(
        self,
        n_estimators: int = 100,
        random_state: int = 42,
        criterion: str = "squared_error",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str = "sqrt",
        **kwargs,
    ):
        super().__init__(
            n_estimators=n_estimators,
            random_state=random_state,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            **kwargs,
        )
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize Extra Trees model."""
        self.model = ExtraTreesRegressor(**self.model_params)


class GBRImputation(BaseImputationModel):
    """Gradient Boosting imputation model."""

    def __init__(
        self,
        n_estimators: int = 100,
        criterion: str = "friedman_mse",
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        loss: str = "squared_error",
        **kwargs,
    ):
        super().__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            loss=loss,
            **kwargs,
        )
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize Gradient Boosting model."""
        self.model = GradientBoostingRegressor(**self.model_params)
