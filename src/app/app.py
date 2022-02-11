from fastapi import FastAPI
from ray import serve

from  src.models.models import MisinformationDetectionModel
from  src.utils.datatypes import Query, Response

app = FastAPI()

@serve.deployment(route_prefix="/app")
@serve.ingress(app)
class MisinformationDetectionApp:
    def __init__(self):
        MisinformationDetectionModel.deploy()
        self.model = MisinformationDetectionModel.get_handle()

    @app.post("/predict", response_model=Response)
    async def predict(self, query: Query):
        prediction = await(self.model.remote(query.data))
        return Response(prediction=prediction)

    @app.get("/")
    def root(self):
        return "Hello, from our app!"