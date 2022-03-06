import logging
from time import perf_counter
from typing import Dict, Any

import ray
from fastapi import FastAPI
from ray import serve

from src.app.model_deployment import MisinformationDetectionModel
from src.utils.datatypes import Query, Response, AppConfig
from src.utils.data import get_label_to_classname_map

app = FastAPI()

@serve.deployment(route_prefix="/app")
@serve.ingress(app)
class MisinformationDetectionApp:
    def __init__(self, config: AppConfig):
        logging.basicConfig(level=logging.INFO)
        self.num_labels = int(config.artifact_name.split('-')[3])
        self.label_to_classname = get_label_to_classname_map(self.num_labels)
        MisinformationDetectionModel.deploy(config=config)
        self.model = MisinformationDetectionModel.get_handle(sync=True)  # TODO figure out why sync=False fails

    @app.post("/predict", response_model=Response)
    async def predict(self, query: Query) -> Response:
        time_start = perf_counter()
        result = await self.model.remote(query.data)
        result = ray.get(result)  # materialize prediction
        time_end = perf_counter()
        server_side_latency_ms = (time_end - time_start) * 1000
        diagnostics = {"server_side_latency_ms": server_side_latency_ms}
        response = Response(
                    id=query.id, 
                    prediction=result.prediction, 
                    predicted_class=self.label_to_classname[result.prediction], 
                    most_similar_examples=result.statements,
                    example_classes=[self.label_to_classname[label] for label in result.statement_labels],
                    example_similarities=result.statement_similarities,
                    diagnostics=diagnostics
                    )
        logging.info(f"Returned response for query {query.id}, latency: {(time_end - time_start) * 1000:.3f} ms")
        return response

    @app.get("/")
    def root(self):
        return "Hello, from our app!"