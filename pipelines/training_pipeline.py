from zenml import pipeline

from steps.ingest_df import ingestdata
from steps.clean_df import clean_data
from steps.model_train import train_model
from steps.evaluation import evaluate_model

@pipeline(enable_cache=False)
def train_pipeline(data_path: str):
    df = ingestdata(data_path)
    clean_data(df)
    train_model(df)
    evaluate_model(df)
    