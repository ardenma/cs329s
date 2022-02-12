import os
import logging
import pathlib
from typing import List

import torch
import numpy as np
from ray import serve

from src.models.embeddings import BOWEmbedding
from src.models.baselines import LogisticRegression

BASEDIR = pathlib.Path(__file__).parent.parent.absolute()
@serve.deployment
class embedding_model:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.model = BOWEmbedding()
        self.model.load(os.path.join(BASEDIR, "saved_models", "embedding_model.pt"))
        self.embedding_size = self.model.get_embedding_size()

    # Takes as input a list of strings and returns a same size list of embeddings
    @serve.batch(batch_wait_timeout_s=0.1)
    async def batch_handler(self, inputs: List[str]) -> List[torch.tensor]:
        # TODO should actual use the embedding model
        batched_embeddings = [self.model(data) for data in inputs]
        logging.info(f"Embedding model called with batch of {len(inputs)}")
        logging.debug(f"Embedding model called with batch of {len(inputs)} inputs: {inputs}, generated corresponding embedding: {batched_embeddings}")
        return batched_embeddings

    async def __call__(self, input: str) -> torch.tensor:
        return await self.batch_handler(input)

@serve.deployment
class prediction_model:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.model = LogisticRegression()
        self.model.load(os.path.join(BASEDIR, "saved_models", "prediction_model.pt"))

    # Takes as input a list of embeddings and returns a same size list of floats
    @serve.batch(batch_wait_timeout_s=0.1)
    async def batch_handler(self, embeddings: List[torch.tensor]) -> List[float]:
        batched_embeddings = torch.vstack(embeddings)
        batched_predictions = self.model(batched_embeddings)[:,0].tolist()
        batched_predictions = list(map(lambda x: float(x), batched_predictions))  # TODO optimize this
        logging.info(f"Prediction model called with batch of {len(embeddings)}")
        logging.debug(f"Prediction model called with batch of {len(embeddings)} embeddings: {embeddings}, generated corresponding prediction: {batched_predictions}")
        return batched_predictions

    async def __call__(self, embedding: torch.tensor) -> List[float]:
        return await self.batch_handler(embedding)

# max_concurrent_queries is optional. By default, if you pass in an async
# function, Ray Serve sets the limit to a high number.
@serve.deployment(max_concurrent_queries=10)
class MisinformationDetectionModel:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)

        embedding_model.deploy()
        prediction_model.deploy()
        
        self.embedding_model = embedding_model.get_handle(sync=False)
        self.prediction_model = prediction_model.get_handle(sync=False)
    
    # This method can be called concurrently!
    async def __call__(self, input: str) -> float:
        embedding = await self.embedding_model.remote(input=input)
        prediction = await self.prediction_model.remote(embedding=embedding)
        return prediction
