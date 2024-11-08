from zenml import step
import pandas as pd

@step
def feature_selector(df:pd.DataFrame)->pd.DataFrame:
    cols=['vehicle_class','engine_size_l','cylinders','transmission','fuel_type','fuel_consumption_comb_l_per_100_km','fuel_consumption_comb_mpg']
    df=df[cols]
    return df