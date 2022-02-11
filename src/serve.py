import ray
from ray import serve

from src.app.app import MisinformationDetectionApp

ray.init(address="auto", namespace="serve")  # Connects to the local ray cluster
serve.start(detached=True)                   # Initialize a ray serve instance
MisinformationDetectionApp.deploy()          # Deploys our application