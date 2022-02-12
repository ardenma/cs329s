import logging

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
        prediction = await self.model.remote(query.data)
        prediction = ray.get(prediction)  # materialize prediction
        response = Response(id=query.id, prediction=prediction)
        return response

    @app.get("/")
    def root(self):
        return "Hello, from our app!"