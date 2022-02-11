from pydantic import BaseModel

class Query(BaseModel):
    id: int
    data: str

class Response(BaseModel):
    id: int
    prediction: float