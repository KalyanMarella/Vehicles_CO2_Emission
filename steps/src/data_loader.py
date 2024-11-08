import pandas as pd
from sqlalchemy import create_engine, exc

class DataLoader:

    def __init__(self,uri:str)->None:

        self.db_uri = uri
        self.engine = create_engine(self.db_uri)
        self.data = None

    def load_data(self,table_name:str)->None:

        """
        This method is to retreive data from the database from table passed
        Args:
            table_name :str
        Returns:
            Upon succesfull retreival,saved it to self.data no returning
        Raise:
            Raise error if any error occurs while retreiving

        """
        query="select * from "+table_name
        try:
            data=pd.read_sql_query(query,self.engine)
            self.data=data
        except exc.SQLAlchemyError as e:
            raise ValueError(f"Failed to execute query: {e}")
        
    def get_data(self)->pd.DataFrame:

        """
        This method is to return saved data(data saved in load_data function)

        Args:
            No arguments just calling the function
        Returns:
            Return saved data 
        Raise:
            Raise error when data is not saved
        """
        
        if self.data is not None:
            return self.data
        else:
            raise ValueError("Data has not been loaded yet. Please call 'load_data' first.")