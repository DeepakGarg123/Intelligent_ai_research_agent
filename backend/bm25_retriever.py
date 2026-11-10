from langchain_community.retrievers import BM25Retriever


def create_bm25_retriever(chunks, k=5):

    retriever = BM25Retriever.from_documents(
        documents=chunks,
        k=k
    )

    return retriever
