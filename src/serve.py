import ray
from ray import serve

from src.app.app import MisinformationDetectionApp
from src.utils.datatypes import AppConfig

def serve_app(detached: bool=False):
    config = AppConfig(
        artifact_name="daily-tree-15-3-labels:v5"
    )

    ray.init(address="auto", namespace="serve")            # Connects to the local ray cluster
    serve.start(detached=detached)                         # Initialize a ray serve instance
    MisinformationDetectionApp.deploy(config=config)       # Deploys our application

if __name__=="__main__":
    serve_app(True)