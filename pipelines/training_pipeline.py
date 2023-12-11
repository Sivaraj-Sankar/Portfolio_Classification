from zenml import pipeline

from steps.ingest_df import ingestdata
from steps.clean_df import clean_data
from steps.model_train import train_model
from steps.evaluation import evaluate_model

@pipeline(enable_cache=False)
def train_pipeline(data_path: str):
    df = ingestdata(data_path)
    X_train, X_test, y_train, y_test = clean_data(df)
    model = train_model(X_train, X_test, y_train, y_test)
    accuracy_score, precision_score = evaluate_model(model, X_test, y_test)
    