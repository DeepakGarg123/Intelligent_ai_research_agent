from backend.llm_provider import get_groq_model

groq = get_groq_model()

response = groq.invoke("What is RAG in Generative AI? Explain in two lines.")

print("Groq Response:")
print(response.content)