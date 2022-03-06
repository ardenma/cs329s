import argparse
import logging
from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from src.serve import serve_app
from src.tests.utils import sample_test_dataset, send_parallel, print_results, find_response

def main(args):
    if args.test_name == "throughput":
        test_throughput(num_queries=args.n, server_endpoint=args.server_endpoint)
    elif args.test_name == "correctness":
        test_correctness(num_queries=args.n, server_endpoint=args.server_endpoint)
    elif args.test_name == "latency_under_load":
        test_latency_under_load(num_queries=args.n, server_endpoint=args.server_endpoint)
    else:
        raise Exception(f"Unknown test name `{args.test_name}`")

def test_throughput(num_queries: int, server_endpoint: str) -> float:
    # Sample the test dataset
    samples = sample_test_dataset(num_samples=num_queries)
    data_samples = [s["data"] for s in samples]
    
    # Send queries in parallel with ray
    test_results = send_parallel(data=data_samples, server_endpoint=server_endpoint)
    throughput = num_queries / test_results['e2e_delay']

    print("\n****************************************************************************************************")
    print(f"Throughput (server @ {server_endpoint}): {throughput:.3f} requests/s.")
    print("****************************************************************************************************\n")

    return throughput
    

def test_latency_under_load(num_queries: int, server_endpoint: str) -> Dict[str, float]:
    # Sample the test dataset
    samples = sample_test_dataset(num_samples=num_queries)
    data_samples = [s["data"] for s in samples]
    
    # Send queries in parallel with ray
    e2e_latencies_ms = []
    server_side_latencies_ms = []
    network_latencies_ms = []
    for i in range(num_queries):
        test_results = send_parallel(data=[data_samples[i]], server_endpoint=server_endpoint)
        e2e_latency_ms = test_results['e2e_delay'] * 1000
        server_side_latency_ms = test_results["results"][0]["diagnostics"]["server_side_latency_ms"]
        network_latency_ms = e2e_latency_ms - server_side_latency_ms

        e2e_latencies_ms.append(e2e_latency_ms)
        server_side_latencies_ms.append(server_side_latency_ms)
        network_latencies_ms.append(network_latency_ms)
    
    mean_e2e_latency_ms = np.mean(e2e_latencies_ms)
    mean_server_side_latency_ms = np.mean(server_side_latencies_ms)
    mean_network_latency_ms = np.mean(network_latencies_ms)

    print("\n****************************************************************************************************")
    print(f"Avg. latency under load (server @ {server_endpoint}):")
    print(f"Avg. end-to-end latency: {mean_e2e_latency_ms:.3f} ms/request.")
    print(f"Avg. server-side inference latency: {mean_server_side_latency_ms:.3f} ms/request.")
    print(f"Avg. network latency: {mean_network_latency_ms:.3f} ms/request.")
    print("****************************************************************************************************\n")

    result_dict = {
        "mean_e2e_latency_ms": mean_e2e_latency_ms, 
        "mean_server_side_latency_ms": mean_server_side_latency_ms,
        "mean_network_latency_ms": mean_network_latency_ms
        }

    return result_dict

def test_correctness(num_queries: int, server_endpoint: str) -> None:
    # Sample the test dataset
    samples = sample_test_dataset(num_samples=num_queries)
    data_samples = [s["data"] for s in samples]

    # Send queries in parallel with ray
    test_results = send_parallel(data=data_samples, server_endpoint=server_endpoint)

    # Print each result
    results = test_results["results"]
    print_results(results)
    
    labels = []
    predictions = []
    for i, sample in enumerate(samples):
        prediction = find_response(results, i)['prediction']
        label = int(sample['label'])
        predictions.append(prediction)
        labels.append(label)
        result = "CORRECT!" if prediction == label else "WRONG!"
        print(f"Query {i}, prediction: {prediction}, true label: {label}, {result}")
    
    print(f"Accuracy: {accuracy_score(labels, predictions)}")
    print(f"F1 Score (weighted) {f1_score(labels, predictions, average='weighted')}")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Test throughput.')
    parser.add_argument('--test_name', type=str)
    parser.add_argument('--server_endpoint', type=str)
    parser.add_argument('-n', type=int, default=10)
    args = parser.parse_args()

    assert args.test_name, "Need to specify a test name using --test_name"

    if not args.server_endpoint:
        logging.warning("No server endpoint specified, running server locally.")
        serve_app(detached=False)
        args.server_endpoint = "127.0.0.1:8000"

    main(args)