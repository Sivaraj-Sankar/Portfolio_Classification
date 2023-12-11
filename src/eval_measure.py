import logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import accuracy_score,precision_score

class Evaluation(ABC):
    """ 
    Abstract Class Defining strategy for evaluation our models
    
    """
    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """ 
        Claculates the scores for the model
        Args:
            y_true: True labels
            y_pred: Predicted labels
        Returns:
            None
        """
        pass
    
class Accuracy_Score(Evaluation):
    """ 
    Evaluation Strategy that uses the Accuracy_Score
    
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        #return super().calculate_scores(y_true, y_pred)
        try:
            logging.info("Calculating Accuracy")
            accuracyR = accuracy_score(y_true,y_pred)
            logging.info("Accuracy: {}".format(accuracyR))
            return accuracyR
        except Exception as e:
            logging.error("Error in calculating the Accuracy Score: {}".format(e))
            raise e
        
class Precision_Score(Evaluation):
    """
    Evaluation Strategy that uses Precision Score
    
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating the Precision Score")
            precisionR = precision_score(y_true, y_pred)
            logging.info("Precision Score: {}".format(precisionR))
            return precisionR
        except Exception as e:
            logging.error("Error in Calculating precision score: {}".format(e))
            raise e 
        
        