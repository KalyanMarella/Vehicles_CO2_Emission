from steps.src.data_loader import DataLoader
from zenml import step
import logging
import pandas as pd

@step
def IngestData(table_name:str,for_predict:bool=False)->pd.DataFrame:
    """
    Ingesting data by passing the table name and giving it to the DataLoader class

    Returns:
        Returns Ingested Data

    Raise:
        Raise error if any error while Ingesting 
        
    """
    try:
        Loader=DataLoader("postgresql+psycopg2://postgres:1234@localhost:5433/postgres")
        Loader.load_data(table_name)
        data=Loader.get_data()
        logging.info("Data Ingested Succesfully")
        print(data.head())
        if for_predict:
            data.drop("co2_emissions_g_per_km",axis=1,inplace=True)
        return data

    except Exception as e:
        logging.error(e)
        raise e
    
if __name__=="__main__":
    print(IngestData("co2_emission_data"))