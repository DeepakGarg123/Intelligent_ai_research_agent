from backend.embeddings import get_embedding_model

embeddings = get_embedding_model()

vector = embeddings.embed_query("Python is a Programming Language")

print("Vector length:" , len(vector))
print("First 5 Values:" ,vector[:5])