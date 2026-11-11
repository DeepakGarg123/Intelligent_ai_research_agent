
import streamlit as st
import tempfile
import shutil
from pathlib import Path

from backend.document_loader import load_document
from backend.chunker import split_documents
from backend.vector_store import create_vector_store
from backend.rag_chain import generate_answer
from backend.sources import extract_document_sources

st.set_page_config(
    page_title="Hybrid RAG Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Hybrid RAG Document Assistant")
st.write("Upload your documents, process them, and ask questions.")

if "upload_dir" not in st.session_state:
    st.session_state.upload_dir = tempfile.mkdtemp(
        prefix="hybrid_rag_"
    )

uploaded_files = st.file_uploader(
    "Upload your documents",
    type=["pdf", "docx", "pptx", "xlsx", "csv", "txt", "md"],
    accept_multiple_files=True
)

if uploaded_files:
    st.write("### Selected documents")

    for file in uploaded_files:
        st.write(f"📄 {file.name}")

if st.button("Process Documents", disabled=not uploaded_files):

    try:
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

        with st.spinner("Creating chunks..."):
            chunks = split_documents(all_documents)

        with st.spinner("Creating vector database..."):
            vector_store = create_vector_store(chunks)

        st.session_state.vector_store = vector_store
        st.session_state.processed_files = [
            file.name for file in uploaded_files
        ]
        st.session_state.chunk_count = len(chunks)

        st.success("Documents processed successfully!")

    except Exception as error:
        st.error(f"Processing failed: {error}")
        st.exception(error)

if "vector_store" in st.session_state:

    st.divider()
    st.subheader("Processing Summary")

    st.write("### Uploaded files")

    for file in st.session_state.processed_files:
        st.write(f"✅ {file}")

    st.write(f"Total chunks created: {st.session_state.chunk_count}")

    st.success("Your documents are ready for questions!")

    st.divider()
    st.header("Ask Questions")

    with st.form("question_form"):

        question = st.text_input(
            "Enter your question",
            placeholder="Ask something about your documents..."
        )

        submitted = st.form_submit_button("Get Answer")

    if submitted and question.strip():

        try:
            with st.spinner("Searching your documents..."):

                results = st.session_state.vector_store.similarity_search_with_score(
                    question,
                    k=5
                )

                context = "\n\n".join(
                    document.page_content
                    for document, score in results
                )

                answer = generate_answer(question, context)
                sources = extract_document_sources(results)

            st.subheader("Answer")
            st.write(answer)

            st.subheader("Document Sources")

            for source in sources:
                st.write(
                    f"📄 {source['file']} | Page: {source['page']}"
                )

        except Exception as error:
            st.error(f"Question processing failed: {error}")
            st.exception(error)
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            