from backend.document_loader import load_document
from backend.chunker import split_documents
from backend.vector_store import create_vector_store

documents = load_document("Deepak_Garg Resume.pdf")
chunks = split_documents(documents)
vector_store = create_vector_store(chunks)

results = vector_store.similarity_search_with_score(
    "What programming Language does Deepak know??",
    k=3
)

for result in results:
    document = result[0]
    score = result[1]
    
    print(document.page_content)
    print(document.metadata)
    