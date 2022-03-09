import os
import logging
import pathlib
from datetime import datetime
from time import perf_counter

import ray
from fastapi import FastAPI
from ray import serve

from backend.app.model_deployment import MisinformationDetectionModel
from backend.utils.data import get_label_to_classname_map
from utils.datatypes import (
    ModelInfoResponse,
    Query,
    Response,
    Feedback,
    PredictionFeedback,
    FeedbackResponse,
    AppConfig,
)

app = FastAPI()

BASEDIR = pathlib.Path(__file__).parent.parent.resolve()
LOGDIR = os.path.join(BASEDIR, "deployment_logs")
if not os.path.exists(LOGDIR):
    os.mkdir(LOGDIR)


@serve.deployment(route_prefix="/app")
@serve.ingress(app)
class MisinformationDetectionApp:
    def __init__(self, config: AppConfig):
        logging.basicConfig(level=logging.INFO)
        self.name = config.artifact_name
        self.num_labels = int(config.artifact_name.split("-")[3])
        self.label_to_classname = get_label_to_classname_map(self.num_labels)
        self.deployment_time = datetime.now()
        MisinformationDetectionModel.deploy(config=config)
        self.model = MisinformationDetectionModel.get_handle(
            sync=True
        )  # TODO figure out why sync=False fails

    @app.post("/predict", response_model=Response)
    async def predict(self, query: Query) -> Response:
        time_start = perf_counter()
        result = await self.model.remote(query.data)
        result = ray.get(result)  # materialize prediction
        time_end = perf_counter()
        server_side_latency_ms = (time_end - time_start) * 1000
        diagnostics = {
            "server_side_latency_ms": server_side_latency_ms,
            "model_name_and_version": self.name,
        }
        response = Response(
            id=query.id,
            prediction=result.prediction,
            predicted_class=self.label_to_classname[result.prediction],
            most_similar_examples=result.statements,
            example_classes=[
                self.label_to_classname[label] for label in result.statement_labels
            ],
            example_similarities=result.statement_similarities,
            diagnostics=diagnostics,
        )
        logging.info(
            f"Returned response for query {query.id}, latency: {(time_end - time_start) * 1000:.3f} ms"
        )
        return response

    @app.post("/feedback", response_model=FeedbackResponse)
    async def log_feedback(self, feedback: Feedback) -> FeedbackResponse:
        with open(
            os.path.join(LOGDIR, f"deployment_log-{self.deployment_time}.txt"), "a+"
        ) as logfile:
            logfile.write(f"({self.name}) {datetime.now()}: {feedback.text_feedback}\n")
        logging.info(f"Received feedback: {feedback.text_feedback}")
        return FeedbackResponse(ack=True)

    @app.post("/prediction_feedback", response_model=FeedbackResponse)
    async def log_prediction_feedback(
        self, feedback: PredictionFeedback
    ) -> FeedbackResponse:
        with open(
            os.path.join(LOGDIR, f"deployment_log-{self.deployment_time}.txt"), "a+"
        ) as logfile:
            logfile.write(
                f"({self.name}) {datetime.now()}: for query '{feedback.query}' predicted '{feedback.predicted_class}' user suggested label '{feedback.user_suggested_class}'\n"
            )
        logging.info(f"Received prediction feedback: {feedback}")
        return FeedbackResponse(ack=True)

    @app.get("/model_info", response_model=ModelInfoResponse)
    async def get_model_info(self) -> ModelInfoResponse:
        if (
            self.name == "daily-tree-15-3-labels:v5"
        ):  # TODO save more metadata with artifacts and fix hardcoding
            response = ModelInfoResponse(
                model_name=self.name.split(":")[0],
                model_version=self.name.split(":")[1],
                model_test_f1_score=0.414,
            )
        else:
            response = ModelInfoResponse(
                model_name=self.name.split(":")[0],
                model_version=self.name.split(":")[1],
            )
        return response

    @app.get("/")
    def root(self):
        return "Hello, from our app!"
