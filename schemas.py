# schemas.py
from pydantic import BaseModel, Field

class AgentRequest(BaseModel):
    question: str
    # Fix : per-session thread scoping. Client should generate one id per
    # conversation/browser session and reuse it across turns so LangGraph
    # loads that thread's history instead of one global growing thread.
    # Falls back to a shared default so existing callers don't break.
    session_id: str = Field(default="default_user_session")

class AgentResponse(BaseModel):
    output: str
    session_id: str