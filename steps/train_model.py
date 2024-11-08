from steps.src.model_building import LinearRegressionModel
from zenml import step
from zenml.logger import get_logger
import mlflow
logger=get_logger(__name__)
import pandas as pd
from sklearn.base import RegressorMixin
from zenml.client import Client
from zenml.integrations.mlflow.experiment_trackers import MLFlowExperimentTracker
from zenml.logger import get_logger
from typing_extensions import Annotated
from typing import Tuple,List
from materializer.custom_materializer import SKLearnModelMaterializer,ListMaterializer

experiment_tracker = Client().active_stack.experiment_tracker

if not experiment_tracker or not isinstance(
    experiment_tracker, MLFlowExperimentTracker
):
    raise RuntimeError(
        "Your active stack needs to contain a MLFlow experiment tracker for "
        "this example to work."
    )

@step(experiment_tracker="mlflow_tracker",
  settings={"experiment_tracker.mlflow": {"experiment_name": "test_name"}},
  enable_cache=False, output_materializers=[SKLearnModelMaterializer, ListMaterializer])
def train_model(x_train:pd.DataFrame,y_train:pd.Series)->Tuple[
    Annotated[RegressorMixin, "model"],
    Annotated[List[str], "predictors"],
]:
    try:
        mlflow.end_run()
        with mlflow.start_run() as run:
            mlflow.sklearn.autolog()  
            model=LinearRegressionModel(x_train,y_train)
            model=model.train()
            logger.info("Model trained successfully")
            predictors=x_train.columns.tolist()
            return model,predictors
    except Exception as e:
        logger.error(e)
        raise e
    