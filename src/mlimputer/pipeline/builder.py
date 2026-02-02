import pandas as pd

from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

from mlimputer.schemas.config import ImputerConfig
from mlimputer.utils.logging import logger
from mlimputer.utils.constants import PipelineStep


@dataclass
class ImputationPipeline:
    """Represents a complete imputation pipeline."""

    steps: List[PipelineStep]
    config: Optional[ImputerConfig]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "ImputationPipeline":
        """Fit the pipeline."""
        logger.info("Starting pipeline fitting")
        data = X.copy()

        for step in self.steps:
            if step.operation == "preprocess":
                data = step.component.fit_transform(data)
            elif step.operation == "impute" and y is not None:
                step.component.fit(data.values, y.values)
            step.is_fitted = True
            logger.debug(f"Fitted step: {step.name}")

        logger.info("Pipeline fitting complete")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data through the pipeline."""
        logger.info("Starting pipeline transformation")
        data = X.copy()

        for step in self.steps:
            if not step.is_fitted:
                raise RuntimeError(f"Step {step.name} is not fitted")

            if step.operation in ["preprocess", "postprocess"]:
                data = step.component.transform(data)
            logger.debug(f"Transformed with step: {step.name}")

        logger.info("Pipeline transformation complete")
        return data
