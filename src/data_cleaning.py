import logging
from abc import ABC, abstractmethod
from typing import Union


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class DataStrategy(ABC):
    """
    Abstract class defining strategy for handling data
    
    """
    
    @abstractmethod
    def handle_data(self,data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass
    
class DataPreProcessingStrategy(DataStrategy):
    """
    Strategy for preprocessing data
    
    """
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame: 
        #return super().handle_data(data)
        """
        Preprocess data
        
        """
        try:
            data['job'].fillna('Missing',inplace=True)
            data['education'].fillna(method='ffill',inplace=True)
            data['contact'].fillna('cellular',inplace=True)
            data['poutcome'].fillna('failure',inplace=True)
            data.drop(columns = ['day_of_week','month'],inplace=True)
            return data
        except Exception as e:
            logging.error("Error in preprocessing data: {}".format(e))
            raise e 
        
class DataEncodingStrategy(DataStrategy):
    """
    Strategy for Encoding Categorical and Feature Engineering the data
    
    """      
    