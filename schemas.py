# schemas.py
from pydantic import BaseModel

class AgentRequest(BaseModel):
    question: str

class AgentResponse(BaseModel):
    output: str