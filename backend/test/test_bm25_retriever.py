from backend.document_loader import load_document
from backend.chunker import split_documents
from backend.bm25_retriever import create_bm25_retriever

documents = load_document("Codewithharry.pdf")

chunks = split_documents(documents)

retriever = create_bm25_retriever(chunks, k=10)

question = input("Enter your query: ")

results = retriever.invoke(question)

for document in results:
    print("\nSource:", document.metadata)
    print("Content:", document.page_content[:500])