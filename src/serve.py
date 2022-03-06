import logging

import ray
from ray import serve

from src.app.app import MisinformationDetectionApp
from src.utils.datatypes import AppConfig

DEFAULT_CONFIG = AppConfig(
    artifact_name="daily-tree-15-3-labels:v5",
    num_embedding_model_replicas=2,
    num_prediction_model_replicas=2
)

def serve_app(config: AppConfig=None, detached: bool=False):
    if not config:
        logging.warning(f"No config specified, using default config:\n{DEFAULT_CONFIG.dict()}")
        config = DEFAULT_CONFIG

    ray.init(address="auto", namespace="serve")            # Connects to the local ray cluster
    serve.start(detached=detached)                         # Initialize a ray serve instance
    MisinformationDetectionApp.deploy(config=config)       # Deploys our application

def redeploy_app(config: AppConfig):
    MisinformationDetectionApp.deploy(config=config)

if __name__=="__main__":
    serve_app(DEFAULT_CONFIG, True)