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
        # Config tells LangGraph which thread history to load/save
        config = {"configurable": {"thread_id": "default_user_session"}}
        
        # Pass the message AND the thread configuration parameter
        response = await agent.ainvoke(
            {"messages": [("user", request.question)]},
            config=config
        )       
        final_message = response["messages"][-1].content
        
        return AgentResponse(output=final_message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent Execution Failure: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "healthy"}