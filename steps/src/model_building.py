import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression


class LinearRegressionModel:
    def __init__(self,x_train:pd.DataFrame,y_train:pd.Series):

        self.x_train=x_train
        self.y_train=y_train
        self.model=LinearRegression()
    
    def train(self)->RegressorMixin:
        self.model.fit(self.x_train,self.y_train)
        return self.model

    
