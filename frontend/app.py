
import streamlit as st
import tempfile
import shutil
from pathlib import Path

from backend.document_loader import load_document
from backend.chunker import split_documents
from backend.vector_store import create_vector_store
from backend.rag_chain import generate_answer
from backend.sources import extract_document_sources
from backend.bm25_retriever import create_bm25_retriever
from backend.hybrid_retriever import reciprocal_rank_fusion
from backend.question_router import route_question
from backend.full_document_search import search_entire_document


st.set_page_config(
    page_title="Hybrid RAG Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Hybrid RAG Document Assistant")
st.write("Upload your documents, process them, and ask questions.")


# Initialize session state

if "upload_dir" not in st.session_state:
    st.session_state.upload_dir = tempfile.mkdtemp(
        prefix="hybrid_rag_"
    )

if "index_build_count" not in st.session_state:
    st.session_state.index_build_count = 0


# Document upload

uploaded_files = st.file_uploader(
    "Upload your documents",
    type=["pdf", "docx", "pptx", "xlsx", "csv", "txt", "md"],
    accept_multiple_files=True
)


if uploaded_files:

    st.write("### Selected documents")

    for file in uploaded_files:
        st.write(f"📄 {file.name}")


# Process documents and build indexes once

if st.button("Process Documents", disabled=not uploaded_files):

    try:

        if Path(st.session_state.upload_dir).exists():
            shutil.rmtree(st.session_state.upload_dir)

        Path(st.session_state.upload_dir).mkdir(
            parents=True,
            exist_ok=True
        )

        all_documents = []

        with st.spinner("Loading documents..."):

            for index, uploaded_file in enumerate(uploaded_files):

                file_path = (
                    Path(st.session_state.upload_dir)
                    / f"{index}_{Path(uploaded_file.name).name}"
                )

                file_path.write_bytes(uploaded_file.getvalue())

                documents = load_document(str(file_path))

                for document in documents:
                    document.metadata["source"] = uploaded_file.name

                all_documents.extend(documents)

        if not all_documents:
            st.error("No readable content found in the uploaded documents.")

        else:

            with st.spinner("Creating chunks..."):
                chunks = split_documents(all_documents)

            if not chunks:
                st.error("No text chunks could be created.")

            else:

                with st.spinner("Creating FAISS vector index..."):
                    vector_store = create_vector_store(chunks)

                with st.spinner("Creating BM25 keyword index..."):
                    bm25_retriever = create_bm25_retriever(
                        chunks,
                        k=20
                    )

                # Save indexes for reuse

                st.session_state.vector_store = vector_store
                st.session_state.bm25_retriever = bm25_retriever
                st.session_state.chunks = chunks

                st.session_state.processed_files = [
                    file.name for file in uploaded_files
                ]

                st.session_state.chunk_count = len(chunks)

                # Count successful index builds

                st.session_state.index_build_count += 1

                st.success("Documents indexed successfully!")

    except Exception as error:

        st.error(f"Processing failed: {error}")
        st.exception(error)


# Display indexing status

if "vector_store" in st.session_state:

    st.divider()
    st.subheader("Processing Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Processed Files",
        len(st.session_state.processed_files)
    )

    col2.metric(
        "Total Chunks",
        st.session_state.chunk_count
    )

    col3.metric(
        "Index Builds",
        st.session_state.index_build_count
    )

    st.write("### Uploaded files")

    for file in st.session_state.processed_files:
        st.write(f"✅ {file}")

    st.success("Your documents are ready for questions!")

    st.divider()
    st.header("Ask Questions")


    # Question input

    with st.form("question_form"):

        question = st.text_input(
            "Enter your question",
            placeholder="Ask something about your documents..."
        )

        submitted = st.form_submit_button("Get Answer")


    if submitted and question.strip():

        try:

            with st.spinner("🧐Understanding your question..."):
                route = route_question(question)

            st.info(f"Selected retrieval strategy: {route}")

            # Hybrid retrieval

            if route == "hybrid":

                with st.spinner("Searching existing indexes..."):

                    # Reuse previously created indexes

                    vector_store = st.session_state.vector_store
                    bm25_retriever = st.session_state.bm25_retriever

                    faiss_results = vector_store.similarity_search(
                        question,
                        k=20
                    )

                    bm25_results = bm25_retriever.invoke(question)

                    results = reciprocal_rank_fusion(
                        faiss_results=faiss_results,
                        bm25_results=bm25_results,
                        k=10
                    )

                if not results:

                    st.warning(
                        "No relevant information found in the uploaded documents."
                    )

                else:

                    context = "\n\n".join(
                        document.page_content
                        for document, score in results
                    )

                    with st.spinner("Generating answer..."):
                        answer = generate_answer(question, context)

                    sources = extract_document_sources(results)

                    st.subheader("Answer")
                    st.markdown(answer)

                    st.subheader("Document Sources")

                    for source in sources:

                        st.write(
                            f"📄 {source['file']} | Page: {source['page']}"
                        )


            # Full-document retrieval

            elif route == "full_document":

                with st.spinner("Searching the entire document..."):

                    results = search_entire_document(
                        st.session_state.chunks,
                        question
                    )

                if not results:

                    st.warning(
                        "No relevant information found in the uploaded documents."
                    )

                else:

                    answer = (
                        "Here are the relevant items found in your documents:\n\n"
                    )

                    for index, item in enumerate(results, start=1):

                        answer += (
                            f"**{index}. {item.get('name', 'Unnamed item')}**\n\n"
                            f"{item.get('description', '')}\n\n"
                            f"Source: {item.get('source', 'Unknown')}\n\n"
                            f"Pages: {', '.join(map(str, item.get('pages', ['Unknown'])))}\n\n"
                            "---\n\n"
                        )

                    st.subheader("Answer")
                    st.markdown(answer)

                    st.subheader("Document Sources")

                    seen_sources = set()

                    for item in results:

                        source = item.get("source", "Unknown")

                        for page in item.get("pages", ["Unknown"]):

                            source_info = (source, page)

                            if source_info not in seen_sources:

                                seen_sources.add(source_info)

                                st.write(
                                    f"📄 {source} | Page: {page}"
                                )

            else:

                st.warning(
                    "Unknown retrieval strategy. Please try the question again."
                )

        except Exception as error:

            st.error(f"Question processing failed: {error}")
            st.exception(error)
