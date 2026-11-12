def reciprocal_rank_fusion(faiss_results, bm25_results, k=10, rrf_k=60):

    fused_scores = {}
    documents = {}

    retrieval_lists = [faiss_results, bm25_results]

    for retrieved_docs in retrieval_lists:

        for rank, document in enumerate(retrieved_docs, start=1):

            metadata = document.metadata

            key = (
                metadata.get("source", ""),
                metadata.get("page", metadata.get("page_label", "")),
                document.page_content
            )

            if key not in fused_scores:
                fused_scores[key] = 0
                documents[key] = document

            score = 1 / (rrf_k + rank)

            fused_scores[key] += score

    ranked_results = sorted(
        fused_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    final_results = []

    for key, score in ranked_results[:k]:
        final_results.append((documents[key], score))

    return final_results


