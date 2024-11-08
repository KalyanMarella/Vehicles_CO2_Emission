from zenml import pipeline
from zenml.logger import get_logger

logger = get_logger(__name__)

from zenml.client import Client

print(Client().active_stack.experiment_tracker.get_tracking_uri())


from zenml import pipeline
from zenml.config import DockerSettings
from zenml.integrations.constants import BENTOML

from steps.bento_builder import bento_builder
from steps.data_splitter import splitter
from steps.deployer import bentoml_model_deployer
from steps.deployment_trigger_step import deployment_trigger
from steps.evaluation import evaluation
from steps.data_ingest import IngestData
from steps.data_preprocess import outlier_handling,category_handling,feature_transform,feature_scaling
#from steps.refine_model import remove_insignificant_vars
from steps.train_model import train_model

docker_settings = DockerSettings(required_integrations=[BENTOML])


@pipeline(enable_cache=False, settings={"docker": docker_settings})
def training_co2_emission():
    """Train a model and deploy it with BentoML."""

    data = IngestData("co2_emission_data")

    x_train,x_test,y_train,y_test = splitter(data)
    
    x_train_after_outlier=outlier_handling(x_train)
    x_test_after_outlier=outlier_handling(x_test)

    train_encode,x_train_after_encoding=category_handling(x_train_after_outlier)
    test_encode,x_test_after_encoding=category_handling(x_test_after_outlier)

    train_transform,x_train_after_transform=feature_transform(x_train_after_encoding)
    test_transform,x_test_after_transform=feature_transform(x_test_after_encoding)

    train_scaler,x_train_after_scaling=feature_scaling(x_train_after_transform)
    test_scaler,x_test_after_scaling=feature_scaling(x_test_after_transform)

    model,predictors = train_model(x_train_after_scaling,y_train)
    r2,rmse=evaluation(model, x_test_after_scaling,y_test)
    
    decision = deployment_trigger(accuracy=r2, min_accuracy=0.85)
    bento = bento_builder(model=model)
    bentoml_model_deployer(bento=bento, deploy_decision=decision) 