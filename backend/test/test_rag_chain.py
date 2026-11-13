
from backend.document_loader import load_document
from backend.chunker import split_documents
from backend.vector_store import create_vector_store
from backend.rag_chain import generate_answer
from backend.sources import extract_document_sources
from backend.bm25_retriever import create_bm25_retriever
from backend.hybrid_retriever import reciprocal_rank_fusion
from backend.question_router import route_question
from backend.full_document_search import search_entire_document


documents = load_document("Codewithharry.pdf")

chunks = split_documents(documents)

question = input("Enter your Query: ")

route = route_question(question)

print("\nSelected Route:", route)


if route == "hybrid":

    print("\nUsing Hybrid Retrieval...")

    vector_store = create_vector_store(chunks)

    bm25_retriever = create_bm25_retriever(chunks, k=20)

    faiss_results = vector_store.similarity_search(
        question, k=20
    )

    bm25_results = bm25_retriever.invoke(question)

    results = reciprocal_rank_fusion(
        faiss_results=faiss_results,
        bm25_results=bm25_results,
        k=10
    )

    context = "\n\n".join(
        document.page_content
        for document, score in results
    )

    answer = generate_answer(question, context)

    sources = extract_document_sources(results)


elif route == "full_document":

    print("\nUsing Full Document Search...")

    results = search_entire_document(chunks, question)

    if not results:

        print("\nNo relevant information found in the document.")
        raise SystemExit

    print("\nExtracted Unique Items:")

    for index, item in enumerate(results, start=1):

        print(f"\nItem {index}:")
        print("Name:", item.get("name"))
        print("Description:", item.get("description"))
        print("Source:", item.get("source"))
        print("Pages:", item.get("pages"))

    context = "\n\n".join(
        f"Name: {item.get('name', '')}\n"
        f"Description: {item.get('description', '')}\n"
        f"Source: {item.get('source', 'Unknown')}\n"
        f"Pages: {item.get('pages', [])}"
        for item in results
    )

    answer = generate_answer(question, context)

    sources = []

    seen_sources = set()

    for item in results:

        source = item.get("source", "Unknown")

        for page in item.get("pages", ["Unknown"]):

            source_info = (source, page)

            if source_info not in seen_sources:

                seen_sources.add(source_info)

                sources.append({
                    "file": source,
                    "page": page
                })


print("\nQuestion:", question)

print("\nAnswer:", answer)

print("\nSources:")

for source in sources:

    print(
        f"File: {source['file']} | Page: {source['page']}"
    )
