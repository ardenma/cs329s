import logging
from typing import List

import numpy as np
from ray import serve
from sklearn.linear_model import LogisticRegression

@serve.deployment
class embedding_model:
    def __init__(self, embedding_size: int):
        logging.basicConfig(level=logging.INFO)
        self.embedding_size = embedding_size
        self.model = None  # TODO load a real model

    # Takes as input a list of strings and returns a same size list of embeddings
    @serve.batch(batch_wait_timeout_s=0.1)
    async def batch_handler(self, inputs: List[str]) -> List[np.array]:
        # TODO should actual use the embedding model
        batched_embeddings = [np.random.rand(self.embedding_size) for _ in range(len(inputs))]
        logging.info(f"Embedding model called with batch of {len(inputs)}")
        logging.debug(f"Embedding model called with batch of {len(inputs)} inputs: {inputs}, generated corresponding embedding: {batched_embeddings}")
        return batched_embeddings

    async def __call__(self, input: str) -> np.array:
        return await self.batch_handler(input)

@serve.deployment
class prediction_model:
    def __init__(self, embedding_size: int):
        logging.basicConfig(level=logging.INFO)
        self.embedding_size = embedding_size
        self.model = LogisticRegression()
        
        # TODO should load a model, using this only for test purposes
        X = np.random.rand(100,embedding_size)
        y = np.random.randint(0,2,100)
        self.model.fit(X,y)

    # Takes as input a list of embeddings and returns a same size list of floats
    @serve.batch(batch_wait_timeout_s=0.1)
    async def batch_handler(self, embeddings: List[np.array]) -> List[float]:
        batched_embeddings = np.vstack(embeddings)
        batched_predictions = self.model.predict_proba(batched_embeddings)[:,1].tolist()
        batched_predictions = list(map(lambda x: float(x), batched_predictions))  # TODO optimize this
        logging.info(f"Prediction model called with batch of {len(embeddings)}")
        logging.debug(f"Prediction model called with batch of {len(embeddings)} embeddings: {embeddings}, generated corresponding prediction: {batched_predictions}")
        return batched_predictions

    async def __call__(self, embedding: np.array) -> List[float]:
        return await self.batch_handler(embedding)

# max_concurrent_queries is optional. By default, if you pass in an async
# function, Ray Serve sets the limit to a high number.
@serve.deployment(max_concurrent_queries=10)
class MisinformationDetectionModel:
    def __init__(self, embedding_size: int=10):
        logging.basicConfig(level=logging.INFO)
        self.embedding_size = embedding_size

        embedding_model.deploy(embedding_size=embedding_size)
        prediction_model.deploy(embedding_size=embedding_size)
        
        self.embedding_model = embedding_model.get_handle(sync=False)
        self.prediction_model = prediction_model.get_handle(sync=False)
    
    # This method can be called concurrently!
    async def __call__(self, input: str) -> float:
        embedding = await self.embedding_model.remote(input=input)
        prediction = await self.prediction_model.remote(embedding=embedding)
        return prediction
