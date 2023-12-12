import numpy as np 
import pandas as pd

from typing import cast
from zenml.config import DockerSettings
from zenml import pipeline, step
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.steps import BaseParameters, Output

from steps.clean_df import clean_data
from steps.evaluation import evaluate_model
from steps.ingest_df import ingestdata
from steps.model_train import train_model


docker_settings = DockerSettings(required_integrations=[MLFLOW])

class DeploymentTriggerConfig(BaseParameters):
    min_accuracy: float = 0.40

@step
def deployment_trigger(
    accuracy: float, 
    config: DeploymentTriggerConfig,
):
    """Implements a model deployment trigger that looks at the input model accuracy and decided if it is good enough to deploy or not"""
    return accuracy > config.min_accuracy

class MLFlowDeploymentLoaderStepParameters(BaseParameters):
    """MLflow deployment getter parameters

    Attributes:
        pipeline_name: name of the pipeline that deployed the MLflow prediction
            server
        step_name: the name of the step that deployed the MLflow prediction
            server
        running: when this flag is set, the step only returns a running service
        model_name: the name of the model that is deployed
    """

    pipeline_name: str
    step_name: str
    running: bool = True

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def continuous_deployment_pipeline(
    data_path: str,
    min_accuracy: float = 0.40,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
):
    df = ingestdata(data_path)
    X_train, X_test, y_train, y_test = clean_data(df)
    model = train_model(X_train, X_test, y_train, y_test)
    accuracy_score, precision_score = evaluate_model(model, X_test, y_test)
    deployment_decision = deployment_trigger(accuracy_score)
    mlflow_model_deployer_step(
        model=model,
        deploy_decision=deployment_decision,
        workers = workers,
        timeout = timeout,
    )
    