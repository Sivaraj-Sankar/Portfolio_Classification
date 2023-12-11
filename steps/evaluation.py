import logging
import pandas as pd
from zenml import step
from typing import Tuple
from typing_extensions import Annotated
from sklearn.base import ClassifierMixin
from src.eval_measure import Evaluation, Accuracy_Score, Precision_Score

@step
def evaluate_model(model: ClassifierMixin,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    ) -> Tuple[
        Annotated[float, "accuracy_score"],
        Annotated[float, "precision_score"]
    ]:
    """
    Evaluates the model on the ingested data.
    Args:
        df: the ingested data
    """
    
    try:
        prediction = model.predict(X_test)
        acc_class = Accuracy_Score()
        accuracy = acc_class.calculate_scores(y_test, prediction)
        
        pre_class = Precision_Score()
        precision = pre_class.calculate_scores(y_test, prediction)
        
        return accuracy, precision
    except Exception as e:
        logging.error("Error in Evaluating model: {}".format(e))
        raise e

