from backend.document_loader import load_document
from backend.chunker import split_documents
from backend.vector_store import create_vector_store
from backend.rag_chain import generate_answer
from backend.sources import extract_document_sources

documents = load_document("Codewithharry.pdf")
chunks = split_documents(documents)
vector_store = create_vector_store(chunks)

question = input("Enter your Query:")

results = vector_store.similarity_search_with_score(question, k=10)

context = "\n\n".join(
    result[0].page_content for result in results
)

for document, score in results:
    print("Source:", document.metadata)
    print("Distance:", score)
    
answer = generate_answer(question , context)

print("\n Question:",question)
print("\n Answer:", answer)

sources = extract_document_sources(results)

print("\nSources:")

for source in sources:   
    print(f"File: {source['file']} | Page: {source['page']}")
    


    
    
    
    
    
    
    
    
    
    
    
    