import argparse
import logging

import ray
from ray import serve

from backend.app.app import MisinformationDetectionApp
from utils.datatypes import AppConfig

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

    host = "0.0.0.0" if args.external_deployment else "127.0.0.1"

    ray.init(address="auto", namespace="serve")            # Connects to the local ray cluster
    serve.start(detached=detached,                         # Initialize a ray serve instance
                http_options={"host": host})          
    MisinformationDetectionApp.deploy(config=config)       # Deploys our application

def redeploy_app(config: AppConfig):
    MisinformationDetectionApp.deploy(config=config)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Start server.')
    parser.add_argument('--artifact_name', default="daily-tree-15-3-labels:v5", type=str)
    parser.add_argument('--num_embedding_model_replicas', default=1, type=int)
    parser.add_argument('--num_prediction_model_replicas', default=1, type=int)
    parser.add_argument('--detached', action='store_true')
    parser.add_argument('--external_deployment', action='store_true')
    args = parser.parse_args()

    config = AppConfig(
        artifact_name=args.artifact_name,
        num_embedding_model_replicas=args.num_embedding_model_replicas,
        num_prediction_model_replicas=args.num_prediction_model_replicas
    )

    serve_app(config, args.detached)
    if not args.detached:
        while(True):
            continue