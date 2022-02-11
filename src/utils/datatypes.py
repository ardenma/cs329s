from pydantic import BaseModel

class Query(BaseModel):
    data: str

class Response(BaseModel):
    prediction: float