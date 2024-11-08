from sklearn.base import RegressorMixin
import pandas as pd
from sklearn.metrics import root_mean_squared_error,r2_score

class RootMeanSquaredError:

    def __init__(self,model:RegressorMixin,x_train:pd.Series,y_train:pd.Series):

        self.model=model
        self.x_train=x_train
        self.y_train=y_train
        self.y_test=None

    def evaluate(self):
        self.y_pred=self.model.predict(self.x_train)
        rmse=root_mean_squared_error(self.y_train,self.y_pred)
        return rmse
    
class R2Score:
    
    def __init__(self,model:RegressorMixin,x_train:pd.Series,y_train:pd.Series):

        self.model=model
        self.x_train=x_train
        self.y_train=y_train
        self.y_test=None

    
    def evaluate(self):
        self.y_pred=self.model.predict(self.x_train)
        r2=r2_score(self.y_train,self.y_pred)
        return r2