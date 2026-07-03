# agent.py
import os
from dotenv import load_dotenv

load_dotenv()

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import trim_messages
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from indexer import _get_chroma_collection, _get_embed_model

# --- Model split (Fix 3) --------------------------------------------------
# AGENT_MODEL drives the ReAct loop itself (tool selection, routing).
# This gets called on every reasoning step, so keep it small/fast.
# SYNTHESIS_MODEL is used only where output quality actually matters
# (summarization, final answer synthesis over retrieved context).
AGENT_MODEL = os.getenv("OLLAMA_AGENT_MODEL", "qwen2.5:1.5b")
SYNTHESIS_MODEL = os.getenv("OLLAMA_SYNTHESIS_MODEL", os.getenv("OLLAMA_MODEL", "qwen3:4b"))

# Max chars a single tool result is allowed to inject back into the
# conversation. Tool output re-enters context on every subsequent turn,
# so this caps how much it can bloat over a long conversation (Fix 2).
MAX_TOOL_OUTPUT_CHARS = 1500

# Token budget for what actually gets sent to Ollama per call, regardless
# of how long the stored conversation has grown (Fix 1).
MAX_CONTEXT_TOKENS = 1500


def _truncate(text: str, limit: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n\n[...truncated, {len(text) - limit} more chars omitted]"


def query_vector_store(query: str) -> str:
    collection = _get_chroma_collection()
    vector_store = ChromaVectorStore(chroma_collection=collection)
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=_get_embed_model())

    local_llm = Ollama(model=SYNTHESIS_MODEL, request_timeout=120.0)
    # similarity_top_k trimmed 3 -> 2, and response_mode="compact" packs
    # retrieved chunks into the fewest possible LLM calls instead of
    # LlamaIndex's default refine loop (which re-invokes the LLM per chunk).
    query_engine = index.as_query_engine(
        llm=local_llm,
        similarity_top_k=2,
        response_mode="compact",
    )

    return str(query_engine.query(query))


@tool
def search_local_docs(query: str) -> str:
    """Searches the indexed local research documents for relevant context."""
    return _truncate(query_vector_store(query))


@tool
def summarize(text: str) -> str:
    """Condenses and summarizes long passages of text or articles into clean bullet points."""
    summary_llm = ChatOllama(model=SYNTHESIS_MODEL, temperature=0.2)
    prompt = f"Please condense and summarize the following text into clear bullet points:\n\n{text}"
    response = summary_llm.invoke(prompt)
    return _truncate(response.content)


def _pre_model_hook(state):
    """
    Runs before every LLM call inside the ReAct loop. Trims the message
    list down to a fixed token budget before it hits Ollama, independent
    of what's actually stored in the checkpoint. Keeps latency flat as a
    conversation thread grows instead of scaling with full history.
    """
    trimmed = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=MAX_CONTEXT_TOKENS,
        start_on="human",
        end_on=("human", "tool"),
        include_system=True,
    )
    return {"llm_input_messages": trimmed}


def get_research_agent():
    llm = ChatOllama(model=AGENT_MODEL, temperature=0)
    tools = [search_local_docs, summarize]

    # Initialize the memory checkpointer
    memory = MemorySaver()

    # Pass the checkpointer and trimming hook to the agent
    return create_react_agent(
        llm,
        tools,
        checkpointer=memory,
        pre_model_hook=_pre_model_hook,
    )