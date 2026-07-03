# app.py
from typing import Any
from fastapi import FastAPI, Depends, HTTPException
from schemas import AgentRequest, AgentResponse
from agent import get_research_agent 

app = FastAPI(
    title="Research Scholar AI Agent",
    description="Local network API serving a ReAct agent framework over ChromaDB",
    version="1.0.0"
)

_agent = get_research_agent()

def get_agent() -> Any:
    return _agent

@app.post("/query", response_model=AgentResponse)
async def query_agent(request: AgentRequest, agent: Any = Depends(get_agent)):
    try:
        # Fix 4: thread_id now comes from the request instead of being
        # hardcoded, so each conversation gets its own isolated history
        # in the checkpointer instead of one thread shared by everyone.
        config = {"configurable": {"thread_id": request.session_id}}

        response = await agent.ainvoke(
            {"messages": [("user", request.question)]},
            config=config
        )
        final_message = response["messages"][-1].content

        return AgentResponse(output=final_message, session_id=request.session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent Execution Failure: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "healthy"}