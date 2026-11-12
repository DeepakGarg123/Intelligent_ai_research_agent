from backend.bm25_retriever import create_bm25_retriever
from backend.document_loader import load_document
from backend.chunker import split_documents
from backend.vector_store import create_vector_store
from backend.rag_chain import generate_answer
from backend.sources import extract_document_sources
from backend.hybrid_retriever import reciprocal_rank_fusion

documents = load_document("Codewithharry.pdf")
chunks = split_documents(documents)
vector_store = create_vector_store(chunks)

question = input("Enter your Query:")

bm25_retriever = create_bm25_retriever(chunks, k=20)

faiss_results = vector_store.similarity_search(question, k=20)

bm25_results = bm25_retriever.invoke(question)

results = reciprocal_rank_fusion(
    faiss_results=faiss_results,
    bm25_results=bm25_results,
    k=10
)

context = "\n\n".join(
    result[0].page_content for result in results
)

for document, score in results:
    print("Source:", document.metadata)
    print("RRF SCORE:", score)
    
from unittest.mock import patch

with patch(
    "backend.llm_provider.get_gemini_model",
    side_effect=Exception("Simulated Gemini failure")
):
    answer = generate_answer(question, context)

print("\n Question:",question)
print("\n Answer:", answer)

sources = extract_document_sources(results)

print("\nSources:")

for source in sources:   
    print(f"File: {source['file']} | Page: {source['page']}")
    