from pipelines.training_pipeline import train_pipeline
from zenml.client import Client


if __name__ == "__main__":
    # Run the Pipeline
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    train_pipeline(data_path="/mnt/c/Users/Sivaraj/OneDrive/Desktop/Final/ZenMLProjects/Project1_Classification/data/bank-full.csv")
    

    