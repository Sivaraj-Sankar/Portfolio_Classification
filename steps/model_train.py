import logging
import pandas as pd 

import mlflow
from zenml import step
from src.model_develop import Model, DecisionClassifierTree
from sklearn.base import ClassifierMixin
# pylint: disable=relative-beyond-top-level
from .config import ModelNameConfig

from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: ModelNameConfig) -> ClassifierMixin:
    """
    Trains the Model on the cleaned Data
    
    Args:
       X_train: pd.DataFrame,
       X_test: pd.DataFrame,
       y_train: pd.Series,
       y_test: pd.Series
    """
    try:
      model = None
      if config.model_name == "DecisionTreeClassifier":
            mlflow.sklearn.autolog()
            model = DecisionClassifierTree()
            trained_model = model.train(X_train,y_train)
            return trained_model
      else:
            raise ValueError("Model {} not supported".format(config.model_name))
    except Exception as e:
       logging.error("Error in Training Model: {}".format(e))
       raise e
    

    
   