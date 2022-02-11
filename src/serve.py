import ray
from ray import serve

from src.app.app import MisinformationDetectionApp

ray.init(address="auto", namespace="serve")  # connects to the local ray cluster
serve.start(detached=True)                  # initialize a ray serve instance

# Deploys our application
MisinformationDetectionApp.deploy()