import logging
import numpy as np
from ray import serve
from sklearn.linear_model import LogisticRegression

@serve.deployment
class embedding_model:
    def __init__(self, embedding_size: int):
        logging.basicConfig(level=logging.INFO)
        self.embedding_size = embedding_size

    def __call__(self, data: str) -> np.array:
        embedding = np.random.rand(self.embedding_size)
        logging.debug(f"Embedding model called with data: {data}, generated corresponding embedding: {embedding}")
        return embedding

@serve.deployment
class prediction_model:
    def __init__(self, embedding_size: int):
        logging.basicConfig(level=logging.INFO)
        self.embedding_size = embedding_size
        self.model = LogisticRegression()
        
        X = np.random.rand(100,embedding_size)
        y = np.random.randint(0,2,100)
        self.model.fit(X,y)

    def __call__(self, embedding: np.array) -> float:
        embedding = np.expand_dims(embedding, axis=0)  # TODO make less brittle
        prediction = self.model.predict_proba(embedding)
        prediction = prediction[0,1]  # getting class 1 prediction from example 0
        logging.debug(f"Prediction model called with embedding: {embedding}, generated corresponding prediction: {prediction}")
        return prediction

# max_concurrent_queries is optional. By default, if you pass in an async
# function, Ray Serve sets the limit to a high number.
@serve.deployment(max_concurrent_queries=10)
class MisinformationDetectionModel:
    def __init__(self, embedding_size: int=10):
        logging.basicConfig(level=logging.INFO)
        self.embedding_size = embedding_size

        embedding_model.deploy(embedding_size=embedding_size)
        prediction_model.deploy(embedding_size=embedding_size)
        
        self.embedding_model = embedding_model.get_handle()
        self.prediction_model = prediction_model.get_handle()

    # This method can be called concurrently!
    async def __call__(self, data):
        embedding = await self.embedding_model.remote(data=data)
        result = await self.prediction_model.remote(embedding=embedding)
        return result

