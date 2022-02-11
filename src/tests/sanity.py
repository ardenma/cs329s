import requests
import ray
from ray import serve

from src.app.app import MisinformationDetectionApp
from src.utils.datatypes import Query

ray.init(address="auto", namespace="serve")  # connects to the local ray cluster
serve.start(detached=False)                  # initialize a ray serve instance

# Deploys our application
MisinformationDetectionApp.deploy()

# Currently just tests 5 queries
for _ in range(5):
    resp = requests.post("http://127.0.0.1:8000/app/predict", data=(Query(data="hey!").json()))
    print(resp.json())