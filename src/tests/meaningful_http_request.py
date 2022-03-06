import requests
from typing import List, Dict, Union

import ray
import numpy as np
from ray import serve

from src.app.app import MisinformationDetectionApp
from src.utils.datatypes import Query
from src.utils.data import LiarDataset

ray.init(address="auto", namespace="serve")  # Connects to the local ray cluster
serve.start(detached=False)                  # Initialize a ray serve instance
MisinformationDetectionApp.deploy()          # Deploys our application

@ray.remote
def send_query(number: int, data: str) -> requests.Response:
    query = Query(id=number, data=data).json()
    resp = requests.post("http://127.0.0.1:8000/app/predict", data=query)
    return resp

def parse_results(results: List[requests.Response]):
    return [result.json() for result in results]

def find_response(responses: List[Dict[str, Union[int, float]]], id: int):
    for response in responses:
        if response["id"] == id:
            return response
    raise Exception(f"id: {id} not found in the list of responses.")

np.random.seed(0)
dataset = LiarDataset("test")
samples = [dataset[int(rand_idx)] for rand_idx in np.random.randint(0, len(dataset), size=10)]

# Let's use Ray to send all queries in parallel
results = ray.get([send_query.remote(i, sample["data"]) for i, sample in enumerate(samples)])
results = parse_results(results)

print("Result returned:")
for result in results:
    print(result)

for i, sample in enumerate(samples):
    prediction = find_response(results, i)['prediction']
    label = int(sample['label'])
    result = "CORRECT!" if round(prediction) == label else "WRONG!"
    print(f"Query {i}, prediction: {prediction}, true label: {label}, {result}")
