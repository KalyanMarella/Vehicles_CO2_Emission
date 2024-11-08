import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler

class FeatureTransformation:
    """"
    This class Implements the Feature Transformation Operations on data 
    Args:
        method : Method or type of transformation to done on feature whether it is a log transformation or sqrt transformation 
    Methods : 
        fit : Initializes the method of transformation on features 
        transform : Implements the Initialized transformation on features
        fit_transform : fit and transformation methods to run in a single code
    
    """
    def __init__(self,method:str="sqrt"):
        """
        Initializing the variable structures for suitable intialization of methods
        """
        self.method=method
        self.transformation={}
    
    def fit(self,df,columns):
        """
        Initializing the method of transformation for each features

        Args: 
            df        :   Data that to be transformed
            columns   :   Columns/Features of data 
        
        """
        for col in columns:
            self.transformation[col]=self.method

    def transform(self,df,columns)->pd.DataFrame:
        """
        Implenting the transformation on features of data initialized by fit method

        Args: 

            df      : Data that to be transformed
            columns : Columns/Features of data

        Returns
            Returns the Transformed data upon Succesfull Transformation

        """
        df_after_feature_transformation=df.copy()
        
        try:
            for col in columns:
                if self.transformation[col]=="log":
                    df_after_feature_transformation[col]=np.log(df[col])
                   
                elif self.transformation[col]=="sqrt":
                    df_after_feature_transformation[col]=np.sqrt(df[col])

            return df_after_feature_transformation

        except Exception as e:
            raise ValueError("Invalid Transformation \n allowed transformations are [log,sqrt]")
            
            
    def fit_transform(self,df,columns)->pd.DataFrame:

        self.fit(df,columns)
        return self.transform(df,columns)

class FeatureScaling:
    """"
    This class Implements the Feature Scaling Operations on data 
    Args:
        method : Method or type of scaling to done on feature,whether it is a MinMax Scaling or Standard Scaling 
    Methods : 
        fit : Initializes the method of scaling on features 
        transform : Implements the Initialized scaling on features
        fit_transform : fit and transformation methods to run in a single code
    
    """
    def __init__(self,method:str="minmax"):
        """
        Initializing the variable structures for suitable intialization of methods

        """
        self.method=method
        self.scalers=None

    def fit(self,df,columns):

        """
        Initializing the method of Scaling for each features

        Args: 
            df        :   Data that to be transformed
            columns   :   Columns/Features of data 
        
        """
        if self.method=="minmax":
            self.scaler=MinMaxScaler()
            self.scaler.fit(df[columns])

        elif self.method=="standard":
            self.scaler=StandardScaler()
            self.scaler.fit(df[columns])
        else:
            raise ValueError("Invalid method of scaling \n allowed methods [standard,minmax]")
        
    def transform(self,df,columns)->pd.DataFrame:
        """
        Implenting the scaling on features of data,initialized by fit method

        Args: 

            df      : Data that to be transformed
            columns : Columns/Features of data

        Returns
            Returns the Transformed data upon Succesfull scaling

        """
        df_after_scaling=df.copy()
        df_after_scaling=self.scaler.transform(df[columns])
        return df_after_scaling
    
    def fit_transform(self,df,columns)->pd.DataFrame:
        self.fit(df,columns)
        df_scaled=self.transform(df,columns)
        df_scaled=pd.DataFrame(df_scaled,columns=columns)
        return df_scaled