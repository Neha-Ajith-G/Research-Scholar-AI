# agent.py
import os
from dotenv import load_dotenv

load_dotenv() 

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from indexer import _get_chroma_collection, _get_embed_model

def query_vector_store(query: str) -> str:
    collection = _get_chroma_collection()
    vector_store = ChromaVectorStore(chroma_collection=collection)
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=_get_embed_model())
    
    model_name = os.getenv("OLLAMA_MODEL", "qwen3:4b")
    local_llm = Ollama(model=model_name, request_timeout=120.0)
    query_engine = index.as_query_engine(llm=local_llm, similarity_top_k=3)
    
    return str(query_engine.query(query))

@tool
def search_local_docs(query: str) -> str:
    """Searches the indexed local research documents for relevant context."""
    return query_vector_store(query)

@tool
def summarize(text: str) -> str:
    """Condenses and summarizes long passages of text or articles into clean bullet points."""
    model_name = os.getenv("OLLAMA_MODEL", "qwen3:4b")
    summary_llm = ChatOllama(model=model_name, temperature=0.2)
    prompt = f"Please condense and summarize the following text into clear bullet points:\n\n{text}"
    response = summary_llm.invoke(prompt)
    return response.content


def get_research_agent():
    model_name = os.getenv("OLLAMA_MODEL", "qwen3:4b")
    llm = ChatOllama(model=model_name, temperature=0)
    tools = [search_local_docs, summarize]
    
    # Initialize the memory checkpointer
    memory = MemorySaver()
    
    # Pass the checkpointer to the agent
    return create_react_agent(llm, tools, checkpointer=memory)