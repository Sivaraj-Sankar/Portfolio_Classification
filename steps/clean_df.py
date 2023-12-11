import logging
import pandas as pd

from zenml import step
from src.data_cleaning import DataCleaning, DataBalancing, DataStrategy, DataPreProcessingStrategy, DataEncodingStrategy, DataDivideStrategy, BalancingAndScaling
from typing_extensions import Annotated
from typing import Tuple

@step
def clean_data(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    """
    Cleans the data and divides it into train and test
    
    Args:
        df: Raw Data
    Returns:
        X_train: Training Data
        X_test: Testing Data
        y_train: Training Labels
        y_test: Testing Labels
    """
    try:
        process_strategy = DataPreProcessingStrategy()
        data_cleaning = DataCleaning(df, process_strategy)
        processed_data = data_cleaning.handle_data()
        
        encode_strategy = DataEncodingStrategy()
        data_cleaning = DataCleaning(processed_data, encode_strategy)
        encoded_data = data_cleaning.handle_data()
        
        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(encoded_data, divide_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        logging.info("Data Cleaning Completed")
        
        balancing_scaling = BalancingAndScaling()
        data_cleaning = DataBalancing(X_train,X_test,y_train,y_test,balancing_scaling)
        
        return X_train, X_test, y_train, y_test
    
    except Exception as e:
        logging.error("Error in 1st Pipeline Step: {}".format(e))
        raise 
        
        

    