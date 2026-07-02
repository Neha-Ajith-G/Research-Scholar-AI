from ollama_client import OllamaClient

llm = OllamaClient()

question = input("Ask: ")

answer = llm.generate(question)

print("\nAnswer:\n")
print(answer)