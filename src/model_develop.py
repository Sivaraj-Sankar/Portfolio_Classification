import logging
from abc import ABC, abstractmethod
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import ClassifierMixin


class Model(ABC):
    """
    Abstract class for all models
    
    """
    @abstractmethod
    def train(self, X_train, y_train):
        """
        Trains the model
        Args:
            X_train: Training Data
            y_train: Training Data
        Returns:
            None
        """
        pass
    
class DecisionClassifierTree(Model):
    
    """
    Decision Tree Classifier
    
    """
    def train(self, X_train, y_train, **kwargs) -> ClassifierMixin:
        """
        Trains the Model
        Args:
            X_train: Training Data
            y_train: Training Data
        Returns:
            None
            
        """
        try:
            decision_clf = DecisionTreeClassifier(**kwargs)
            model = decision_clf.fit(X_train,y_train)
            #self.model.fit(X_train,y_train)
            logging.info("Model Training Completed")
            return model
            
        except Exception as e:
            logging.error("Error in training model: {}".format(e))
            raise e
        
    