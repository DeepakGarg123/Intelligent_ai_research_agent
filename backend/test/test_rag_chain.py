from backend.document_loader import load_document
from backend.chunker import split_documents
from backend.vector_store import create_vector_store
from backend.rag_chain import generate_answer

documents = load_document("Deepak_Garg Resume.pdf")
chunks = split_documents(documents)
vector_store = create_vector_store(chunks)

question = "What programming language does Deepak know?"

results = vector_store.similarity_search_with_score(question, k=3)

context = "\n\n".join(
    result[0].page_content for result in results
)

for document, score in results:
    print("Source:", document.metadata)
    print("Distance:", score)
    
answer = generate_answer(question , context)

print("\n Question:",question)
print("\n Answer:", answer[0]['text'])
    
    
    
    
    
    
    
    
    
    
    
    
    