import requests
from time import perf_counter
from typing import List, Dict, Any, Union

import ray

from src.utils.datatypes import Query

def find_response(responses: List[Dict[str, Union[int, float]]], id: int):
    for response in responses:
        if response["id"] == id:
            return response
    raise Exception(f"id: {id} not found in the list of responses.")

def parse_results(results: List[requests.Response]) -> List[Dict[str, Any]]:
    return [result.json() for result in results]

def print_results(results: List[Dict[str, Any]]) -> None:
    print("Result returned:")
    for result in results:
        print(result)

@ray.remote
def send_query(query_id: int, data: str="asdfjkl", server_endpoint: str="127.0.0.1:8000") -> requests.Response:
    query = Query(id=query_id, data=data).json()
    resp = requests.post(f"http://{server_endpoint}/app/predict", data=query)
    return resp

def send_parallel(data: List[str], server_endpoint: str="127.0.0.1:8000"):
    # Let's use Ray to send all queries in parallel
    time_start = perf_counter()
    results = ray.get([send_query.remote(i, d, server_endpoint) for i, d in enumerate(data)])
    time_end = perf_counter()
    results = parse_results(results)

    return {f"e2e_delay": time_end - time_start, "results": results}


    