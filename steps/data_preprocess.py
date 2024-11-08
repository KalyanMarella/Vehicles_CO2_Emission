from zenml import step
from zenml.logger import get_logger
from steps.src.data_process import CategoricalEncoder,OutlierHandling
from steps.src.feature_engineering import FeatureScaling,FeatureTransformation
logger=get_logger(__name__)
from typing_extensions import Annotated
from typing import Tuple
import pandas as pd
import numpy as np
from zenml.client import Client

@step(enable_cache=False)
def outlier_handling(df:pd.DataFrame)->pd.DataFrame:
    try:
        outlier=OutlierHandling()
        columns=["engine_size_l",
        "fuel_consumption_comb_l_per_100_km",
        "fuel_consumption_comb_mpg"
        ]
        data_after_outlier_process=outlier.fit_transform(df,columns)
        logger.info("Outlier handling completed successfully")
        return data_after_outlier_process
    except Exception as e:
        logger.error(e)
        raise e

@step(enable_cache=False)
def category_handling(df:pd.DataFrame)->Tuple[
    Annotated[CategoricalEncoder,"encoder_handling"],
    Annotated[pd.DataFrame,"data_after_encode"]
]:
    try:
        encoder_handling=CategoricalEncoder(method="label")
        data_after_encode=encoder_handling.fit_transform(df,columns=["vehicle_class","transmission","fuel_type"])
        logger.info("Categorical data encoded successfully")
        return encoder_handling,data_after_encode
    except Exception as e:
        logger.error(e)
        raise e

@step(enable_cache=False)
def feature_transform(df:pd.DataFrame)->Tuple[
    Annotated[FeatureTransformation,"transformer"],
    Annotated[pd.DataFrame,"data_after_transform"]
]:

    try:
        transformer=FeatureTransformation(method="log")
        df_after_transform=transformer.fit_transform(df,columns=["fuel_consumption_comb_mpg"])
        logger.info("Feature Transformation done successfully")
        return transformer,df_after_transform
    
    except Exception as e:
        logger.error(e)
        raise e

@step(enable_cache=False)
def feature_scaling(df:pd.DataFrame)->Tuple[
    Annotated[FeatureScaling,"scaler"],
    Annotated[pd.DataFrame,"data_after_scaling"]
]:
    try:
        scaler=FeatureScaling("minmax")
        df_after_scaling=scaler.fit_transform(df,list(df.columns))
        logger.info("Feature Scaling done successfully")
        return scaler,df_after_scaling
    except Exception as e:
        logger.error(e)
        raise e
        

@step()
def category_handling_test(test_data: pd.DataFrame) -> pd.DataFrame:
    artifact = Client().get_artifact_version('8c5ce82a-7978-47ae-a1bf-f90054dd27c3')
    loaded_encoder = artifact.load()  # Load the encoder
    
    def safe_transform(encoder,values, unknown_value=-1):
        return np.array([encoder.transform([val])[0] if val in encoder.classes_ else unknown_value for val in values])

    transformed_test_data = test_data.copy()
    for column in ["vehicle_class", "transmission", "fuel_type"]:
        transformed_test_data[column] = safe_transform(loaded_encoder.encoders[column],test_data[column])

    return transformed_test_data


@step()
def feature_transform_test(test_data:pd.DataFrame)->pd.DataFrame:
    artifact = Client().get_artifact_version('2a5e493a-4588-4c9c-959c-46b6445af272')
    loaded_transformer = artifact.load()  
    transformed_test_data = loaded_transformer.transform(test_data,columns=["fuel_consumption_comb_mpg"])
    return transformed_test_data

@step()
def feature_scaling_test(test_data:pd.DataFrame)->pd.DataFrame:
    artifact = Client().get_artifact_version('00e50cd8-9009-4a6f-8bba-0a753448a7a3')
    loaded_scaler = artifact.load()  
    columns=['vehicle_class','engine_size_l','cylinders','transmission','fuel_type','fuel_consumption_comb_l_per_100_km','fuel_consumption_comb_mpg']
    transformed_test_data = loaded_scaler.transform(test_data,columns)
    transformed_test_data=pd.DataFrame(transformed_test_data,columns=columns)
    return transformed_test_data