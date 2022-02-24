import logging
from time import perf_counter

import ray
from fastapi import FastAPI
from ray import serve

from  src.app.model_deployment import MisinformationDetectionModel
from  src.utils.datatypes import Query, Response

app = FastAPI()

@serve.deployment(route_prefix="/app")
@serve.ingress(app)
class MisinformationDetectionApp:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        MisinformationDetectionModel.deploy()
        self.model = MisinformationDetectionModel.get_handle(sync=True)  # TODO figure out why sync=False fails

    @app.post("/predict", response_model=Response)
    async def predict(self, query: Query) -> Response:
        time_start = perf_counter()
        prediction = await self.model.remote(query.data)
        prediction = ray.get(prediction)  # materialize prediction
        response = Response(id=query.id, prediction=prediction)
        time_end = perf_counter()
        logging.info(f"Returned response for query {query.id}, latency: {(time_end - time_start) * 1000:.3f} ms")
        return response

    @app.get("/")
    def root(self):
        return "Hello, from our app!"