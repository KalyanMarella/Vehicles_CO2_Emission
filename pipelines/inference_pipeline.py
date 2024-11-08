from zenml import pipeline
from zenml.config import DockerSettings
from zenml.integrations.constants import BENTOML
import requests
import pandas as pd
from steps.data_ingest import IngestData
from steps.predict_step import predictor
from steps.prediction_service_loader_step import bentoml_prediction_service_loader
from steps.data_preprocess import outlier_handling,category_handling_test,feature_scaling_test,feature_transform_test
from steps.feature import feature_selector
from steps.data_drift import data_drift_detector
from steps.model_drift import model_drift_detector
from steps.baseline_data import load_baseline_x_data,load_baseline_y_data
from steps.model_drift import model_drift_detector

docker_settings = DockerSettings(required_integrations=[BENTOML])


@pipeline(settings={"docker": docker_settings})
def inference_co2_emission(
    model_name: str, pipeline_name: str, step_name: str
):
    """Perform inference with a model deployed through BentoML.

    Args:
        pipeline_name: The name of the pipeline that deployed the model.
        step_name: The name of the step that deployed the model.
        model_name: The name of the model that was deployed.
    """
    inference_data = IngestData(table_name="co2_emission_data", for_predict=True)
    baseline_df=load_baseline_x_data()
    baseline_df_y=load_baseline_y_data()
    data_after_selector=feature_selector(inference_data)
    df_after_outlier = outlier_handling(data_after_selector)
    df_after_encode = category_handling_test(df_after_outlier)
    df_after_transform = feature_transform_test(df_after_encode) 
    df_after_scaling = feature_scaling_test(df_after_transform)

    json_code_data,html_code_data=data_drift_detector(inference_data=df_after_scaling, baseline_data=baseline_df)

    prediction_service = bentoml_prediction_service_loader(
        model_name=model_name, pipeline_name=pipeline_name, step_name=step_name
    )
    predictions = predictor(inference_data=df_after_scaling, service=prediction_service)

    json_code_model,html_code_model=model_drift_detector(reference_data=baseline_df_y,current_data=predictions)
