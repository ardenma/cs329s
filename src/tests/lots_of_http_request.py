import requests
from typing import List

import ray
from ray import serve

from src.serve import serve_app
from src.utils.datatypes import Query

# Start serving the app
serve_app(detached=False)

@ray.remote
def send_query(number: int) -> requests.Response:
    query = Query(id=number, data="hey").json()
    resp = requests.post("http://127.0.0.1:8000/app/predict", data=query)
    return resp

def parse_results(results: List[requests.Response]):
    return [result.json() for result in results]

# Let's use Ray to send all queries in parallel
results = ray.get([send_query.remote(i) for i in range(100)])
results = parse_results(results)
print("Result returned:")
for result in results:
    print(result)