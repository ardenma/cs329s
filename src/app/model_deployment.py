import logging
import pathlib
from typing import List

import torch
import faiss
import numpy as np
from ray import serve

from src.models.distilbert import DistilBertForSequenceEmbedding
from src.models.voting import WeightedMajorityVoter
from src.utils.data import LiarDataset
from src.utils.artifacts import download_model_artifact, download_index_artifact
from src.utils.index import cache_index, load_index
from src.utils.datatypes import PredictionResult

BASEDIR = pathlib.Path(__file__).parent.parent.absolute()
@serve.deployment
class embedding_model:
    def __init__(self, artifact_name: str=None):
        logging.basicConfig(level=logging.INFO)
        self.model = DistilBertForSequenceEmbedding()
        
        assert artifact_name is not None, "Need to specify an artifact name."
        self.model_name = artifact_name
        self.num_labels = int(artifact_name.split('-')[3]) # TODO save this info with model
        self.model_path = download_model_artifact(artifact_name)
        self.model.load(self.model_path)
        self.embedding_size = self.model.get_embedding_size()
        logging.info(f"Embedding model loaded from artifact '{artifact_name}'.")

    # Takes as input a list of strings and returns a same size list of embeddings
    @serve.batch(batch_wait_timeout_s=0.1)
    async def batch_handler(self, inputs: List[str]) -> List[torch.tensor]:
        with torch.no_grad():
            batched_embeddings = [self.model(data) for data in inputs]
        logging.info(f"Embedding model called with batch of {len(inputs)}")
        logging.debug(f"Embedding model called with batch of {len(inputs)} inputs: {inputs}, generated corresponding embedding: {batched_embeddings}")
        return batched_embeddings

    async def __call__(self, input: str) -> torch.tensor:
        return await self.batch_handler(input)


@serve.deployment
class prediction_model:
    def __init__(self, artifact_name: str=None, K: int=3):
        logging.basicConfig(level=logging.INFO)

        assert artifact_name is not None, "Need to specify an artifact name."
        self.model_name = artifact_name
        self.num_labels = int(artifact_name.split('-')[3])  # TODO save this info with model
        self.id_map = LiarDataset("train", num_labels=self.num_labels).get_id_map()
        self.K = K
        
        # If index isn't already cached on disk, build one and cache it
        index_path = download_index_artifact(artifact_name)
        self.index = faiss.read_index(index_path)

        logging.info(f"Index for prediction model loaded from artifact '{artifact_name}'.")

        self.voter = WeightedMajorityVoter()


    # Takes as input a list of embeddings and returns a same size list of int label predictions
    @serve.batch(batch_wait_timeout_s=0.1)
    async def batch_handler(self, embeddings: List[torch.tensor]) -> List[PredictionResult]:
        # Batch together embeddings
        batched_embeddings = torch.vstack(embeddings)

        # Compute similarities with IP index searech
        batched_similarities, batched_ids = self.index.search(batched_embeddings.cpu().numpy(), self.K)

        # Vote on prediction
        votes = [list(map(self.get_label_from_id, ids)) for ids in batched_ids]
        batched_predictions = self.voter(votes, batched_similarities)

        logging.info(f"Prediction model called with batch of {len(embeddings)}")
        logging.debug(f"Prediction model called with batch of {len(embeddings)} embeddings: {embeddings}, generated corresponding prediction: {batched_predictions}")

        batched_results = [
            PredictionResult(
                prediction=pred, 
                statements=list(map(self.get_statement_from_id, ids)), 
                statement_ids=list(ids), 
                statement_labels=list(map(self.get_label_from_id, ids)),
                statement_similarities=list(similarities)
            ) for pred, ids, similarities in zip(batched_predictions, batched_ids, batched_similarities)]

        return batched_results
    
    def get_statement_from_id(self, i: int):
        return self.id_map[i]["statement"]

    def get_label_from_id(self, i: int):
        return self.id_map[i]["label"]

    async def __call__(self, embedding: torch.tensor) -> List[int]:
        return await self.batch_handler(embedding)

# max_concurrent_queries is optional. By default, if you pass in an async
# function, Ray Serve sets the limit to a high number.
@serve.deployment(max_concurrent_queries=10)
class MisinformationDetectionModel:
    def __init__(self, artifact_name: str=None):
        logging.basicConfig(level=logging.INFO)

        embedding_model.deploy(artifact_name=artifact_name)
        prediction_model.deploy(artifact_name=artifact_name)
        
        self.embedding_model = embedding_model.get_handle(sync=False)
        self.prediction_model = prediction_model.get_handle(sync=False)
    
    # This method can be called concurrently!
    async def __call__(self, input: str) -> PredictionResult:
        embedding = await self.embedding_model.remote(input=input)
        result = await self.prediction_model.remote(embedding=embedding)
        return result
