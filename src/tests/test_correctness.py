import argparse
import logging

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from src.serve import serve_app
from src.utils.data import LiarDataset
from src.tests.utils import send_parallel, print_results, find_response


def main(args):
    np.random.seed(0)
    dataset = LiarDataset("test")
    samples = [dataset[int(rand_idx)] for rand_idx in np.random.randint(0, len(dataset), size=args.n)]
    data_samples = [s["data"] for s in samples]

    # Send queries in parallel with ray
    test_results = send_parallel(data=data_samples, server_endpoint=args.server_endpoint)

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
    parser.add_argument('--server_endpoint', type=str)
    parser.add_argument('-n', type=int, default=10)
    args = parser.parse_args()

    if not args.server_endpoint:
        logging.warning("No server endpoint specified, running server locally.")
        serve_app(detached=False)
        args.server_endpoint = "127.0.0.1:8000"

    main(args)