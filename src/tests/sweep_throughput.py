import argparse
import multiprocessing

import ray
from ray import serve
from multiprocessing import cpu_count

from src.serve import serve_app, redeploy_app
from src.utils.datatypes import AppConfig
from src.tests.test import test_throughput

def main(args):    

    ray.init(address="auto", namespace="serve")            # Connects to the local ray cluster
    serve.start(detached=False)                            # Initialize a ray serve instance
    CPU_COUNT = int(ray.available_resources()["CPU"])
    print(f"CPUs available in ray cluster: {CPU_COUNT}")
    MAX_REPLICA_COUNT = CPU_COUNT - 4                      # Buffer for other ray processes
    search_space = [{"num_embedding_model_replicas": i, "num_prediction_model_replicas": MAX_REPLICA_COUNT - i} for i in range(1, MAX_REPLICA_COUNT)]

    results = []

    for params in search_space:
        config = AppConfig(
            artifact_name="daily-tree-15-3-labels:v5",
            num_embedding_model_replicas=params["num_embedding_model_replicas"],
            num_prediction_model_replicas=params["num_prediction_model_replicas"]
        )
        redeploy_app(config=config)
        throughput = test_throughput(num_queries=args.n, server_endpoint="127.0.0.1:8000")
        result = params
        result["throughput"] = throughput
        results.append(result)

    best_result = None
    best_throughput = 0
    for result in results:
        if result["throughput"] > best_throughput:
            best_throughput = result["throughput"]
            best_result = result

    print("\n****************************************************************************************************")
    print("THROUGHPUT SWEEP RESULTS:")
    for r in results:
        print(r)
    print("BEST RESULT:")
    print(best_result)
    print("****************************************************************************************************\n")




if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Test throughput.')
    parser.add_argument('-n', type=int, default=100)
    args = parser.parse_args()

    main(args)