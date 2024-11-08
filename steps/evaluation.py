import pandas as pd
from zenml import step
from sklearn.base import RegressorMixin
from steps.src.model_evaluation import R2Score,RootMeanSquaredError
from zenml.logger import get_logger
logger=get_logger(__name__)
from typing import List,Union
from typing import Tuple
from typing_extensions import Annotated

@step(enable_cache=False)
def evaluation(model:RegressorMixin,x_train:pd.DataFrame,y_train:pd.Series)->Tuple[
    Annotated[float,"r2"],
    Annotated[float,"rmse"]
]:
    try:
        r2=R2Score(model,x_train,y_train).evaluate()
        rmse=RootMeanSquaredError(model,x_train,y_train).evaluate()
        logger.info("Model Evaluated Succesfully")
        print(r2,rmse)
        return r2,rmse

    except Exception as e:
        logger.error(e)
        raise e