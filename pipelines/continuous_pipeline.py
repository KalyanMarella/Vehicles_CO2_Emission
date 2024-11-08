from zenml import pipeline
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step

from steps.data_splitter import splitter
from steps.evaluation import evaluation
from steps.data_ingest import IngestData
from steps.data_preprocess import category_handling, feature_scaling,feature_transform,outlier_handling

# from steps.refine_model import remove_insignificant_vars
from steps.train_model import train_model

docker_settings = DockerSettings(required_integrations=[MLFLOW]) 


@pipeline(enable_cache=False, settings={"docker": docker_settings})
def continuous_deployment_pipeline(
    min_accuracy: float = 0.85,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
):
    df = IngestData("co2_emission_data")
    x_train, x_test, y_train, y_test = splitter(df)

    x_train_after_outlier = outlier_handling(x_train)
    x_test_after_outlier = outlier_handling(x_test)

    x_train_after_encode=category_handling(x_train_after_outlier)
    x_test_after_encode=category_handling(x_test_after_outlier)

    x_train_after_transform=feature_transform(x_train_after_encode)
    x_test_after_transform=feature_transform(x_test_after_encode)


    x_train_after_scaling=feature_transform(x_train_after_transform)
    x_test_after_scaling=feature_transform(x_test_after_transform)

    model,predictors = train_model(x_train_after_scaling, y_train) 

    mlflow_model_deployer_step(
        model=model,
        workers=workers,
        timeout=timeout,
    )

