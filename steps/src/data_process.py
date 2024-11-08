from abc import ABC,abstractmethod
from typing import List
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder,OneHotEncoder

class CategoricalEncoder:
    """
    This class is for Handling the Categorical Dataset
    Implemented OneHot Encoding,Ordinal Encoding,Label Encoding strategies

    Args:
        Method : Type of strategy to Implement 

    """
    def __init__(self,method="onehot"):

        self.method=method
        self.encoders={}

    def fit(self,df,columns)->None:

        """
        Initializes the appropriate strategy of encoding according  to the method passed

        """

        if self.method.lower=="onehot":
            for col in columns:
                self.encoders[col]=OneHotEncoder(sparse=False)
                self.encoders[col].fit(df[col])


        elif self.method=="label":
            for col in columns:
                self.encoders[col]=LabelEncoder()
                self.encoders[col].fit(df[col])
        
        elif self.method=="ordinal":
            for col in columns:
                self.encoders[col]=OrdinalEncoder()
                self.encoders[col].fit(df[col])
        else:
            raise ValueError("Invalid method \n Supported methods [onehot,ordinal,label]")
        
    def transform(self,df,columns)->pd.DataFrame:

        """"
        Implements the Initialized strategy column wise and transformation of categorical to numericals happens

        Args:
            df : data to be encoded
            columns : Features of the data to encode

        Returns: 
            Returns encoded data

        """
        df_after_encoding=df.copy()
        for col in columns:
            if self.method=="onehot":
                after_encoding=self.encoders[col].transform[df[col]]
                after_encoding=pd.DataFrame(after_encoding,columns=self.encoders[col].get_feature_names_out([col]))
                df_after_encoding=pd.concat([df_after_encoding,after_encoding],axis=1)
                df_after_encoding.drop(col,axis=1,inplace=True)
            else:
                df_after_encoding[col]=self.encoders[col].transform(df[col])
        return df_after_encoding
        
    def fit_transform(self,df,columns)->pd.DataFrame:
        self.fit(df,columns)
        return self.transform(df,columns)
    
class OutlierHandling:
    """
    This class handles the outlier in tha data by imputing the outliers with the medain of the data

    Args: multiplier : 1.5 

    Methods : 
        fit : Fit method calculates the median and outliers in the features of the data
        transform: Transform method imputes the median at outliers
        fit_transform : fit_transform handles both fit and transform by single command 

    Returns: Returns Outlier free data 
    """

    def __init__(self,multiplier:float=1.5):
        """
        Intializing the variables to handle medians and outliers

        Args: multiplier 
        """
        self.multiplier=multiplier
        self.iqr_bounds={}
        self.medians={}
        self.outliers=pd.DataFrame()
    
    def fit(self,df,columns)->None:

        """
        Caluclates the median of each column of the data 

        Points out the outlier by using the method Inter Quartile Range
        
        """
        for col in columns:
            self.medians[col]=df[col].median()
            q1=df[col].quantile(0.25)
            q3=df[col].quantile(0.75)
            iqr=q3-q1
            lower_bound=q1-self.multiplier*iqr
            upper_bound=q3+self.multiplier*iqr
            self.iqr_bounds[col]=(lower_bound,upper_bound)
    
    def transform(self,df,columns)->pd.DataFrame:
        """
        Imputes the median at the outliers

        Args:
            df : Data in which Outliers has to be detect
            columns : Features of the data 

        Returns :
            Returns the outlier free data after imputation
        """
        df_transformed=df.copy()
        
        for col in columns:
            outliers = df[(df[col] < self.iqr_bounds[col][0]) | (df[col] > self.iqr_bounds[col][1])]
            self.outliers = pd.concat([self.outliers, outliers])
            df_transformed[col] = np.where((df[col] < self.iqr_bounds[col][0]) | (df[col] > self.iqr_bounds[col][1]), self.medians[col], df[col])
        return df_transformed
    
    def fit_transform(self,df,columns)->pd.DataFrame:

        self.fit(df,columns)
        return self.transform(df,columns)