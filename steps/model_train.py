import logging
import pandas as pd 

from zenml import step


@step
def train_model(df: pd.DataFrame) -> None:
    """
    Trains the Model on the cleaned Data
    
    Args:
       df: the ingested data
    """
    pass