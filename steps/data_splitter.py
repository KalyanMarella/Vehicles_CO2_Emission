from zenml import step
from typing import Tuple
from typing_extensions import Annotated
from typing import Union
import pandas as pd
from sklearn.model_selection import train_test_split
from zenml.logger import get_logger
logger=get_logger(__name__)

@step(enable_cache=False)
def splitter(df:pd.DataFrame)->Tuple[
    Annotated[pd.DataFrame, "x_train"],
    Annotated[pd.DataFrame, "x_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    try:
        final_features=['vehicle_class','engine_size_l','cylinders','transmission','fuel_type','fuel_consumption_comb_l_per_100_km','fuel_consumption_comb_mpg']
        x_train,x_test,y_train,y_test=train_test_split(df[final_features],df["co2_emissions_g_per_km"],test_size=0.3)
        logger.info("Data Splitting done Successfully")
        return x_train,x_test,y_train,y_test
    except Exception as e:
        logger.error(e)
        raise e