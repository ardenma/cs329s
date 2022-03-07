import argparse
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
        logging.warning(f"No config specified, using default config.")
        config = DEFAULT_CONFIG
    logging.info(f"config: {config.dict()}")

    ray.init(address="auto", namespace="serve")            # Connects to the local ray cluster
    serve.start(detached=detached)                         # Initialize a ray serve instance
    MisinformationDetectionApp.deploy(config=config)       # Deploys our application

def redeploy_app(config: AppConfig):
    MisinformationDetectionApp.deploy(config=config)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Start server.')
    parser.add_argument('--artifact_name', default="daily-tree-15-3-labels:v5", type=str)
    parser.add_argument('--num_embedding_model_replicas', default=1, type=int)
    parser.add_argument('--num_prediction_model_replicas', default=1, type=int)
    args = parser.parse_args()

    config = AppConfig(
        artifact_name=args.artifact_name,
        num_embedding_model_replicas=args.num_embedding_model_replicas,
        num_prediction_model_replicas=args.num_prediction_model_replicas
    )

    serve_app(config, True)