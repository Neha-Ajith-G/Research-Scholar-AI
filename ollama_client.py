import os
from ollama import chat
from ollama import ResponseError
 
class OllamaConnectionError(Exception):
    """Raised when Ollama isn't reachable or the model isn't available.
    Distinct from ollama's own exceptions so callers (FastAPI routes,
    Streamlit UI) can catch one thing and show a clean error message,
    instead of leaking library-specific exception types up the stack."""
 
 
class OllamaClient:
    def __init__(self, model: str | None = None):
        # Falls back to an env var so the model is configurable without
        # code changes
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
 
    def generate(self, prompt: str) -> str:
        try:
            response = chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
        except ResponseError as exc:
            # Most common case: model not pulled yet
            if getattr(exc, "status_code", None) == 404:
                raise OllamaConnectionError(
                    f"Model '{self.model}' isn't available. "
                    f"Run: ollama pull {self.model}"
                ) from exc
            raise OllamaConnectionError(f"Ollama returned an error: {exc}") from exc
        except ConnectionError as exc:
            raise OllamaConnectionError(
                "Can't reach Ollama. Is it running? Start it with: ollama serve"
            ) from exc
 
        return response.message.content