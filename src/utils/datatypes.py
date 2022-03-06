from typing import List, Dict, Any
from pydantic import BaseModel

class Query(BaseModel):
    id: int
    data: str

class Response(BaseModel):
    id: int
    prediction: int
    predicted_class: str
    most_similar_examples: List[str]
    example_classes: List[str]
    example_similarities: List[float]
    diagnostics: Dict[str, Any]

class PredictionResult(BaseModel):
    prediction: int
    statements: List[str]
    statement_ids: List[int]
    statement_labels: List[int]
    statement_similarities: List[float]

class AppConfig(BaseModel):
    artifact_name: str
    num_embedding_model_replicas: int
    num_prediction_model_replicas: int