import argparse
import logging

import numpy as np

from src.serve import serve_app
from src.utils.data import LiarDataset
from src.tests.utils import send_parallel


def main(args):
    # Sample the test dataset
    np.random.seed(0)
    dataset = LiarDataset("test")
    samples = [dataset[int(rand_idx)]["data"] for rand_idx in np.random.randint(0, len(dataset), size=args.n)]
    
    # Send queries in parallel with ray
    test_results = send_parallel(data=samples, server_endpoint=args.server_endpoint)

    print("\n****************************************************************************************************")
    print(f"Throughput (server @ {args.server_endpoint}): {args.n / (test_results['e2e_delay']):.3f} requests/s.")
    print("****************************************************************************************************\n")

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