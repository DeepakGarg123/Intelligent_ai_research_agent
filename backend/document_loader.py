from pathlib import Path
from langchain_community.document_loaders import(
    PyPDFLoader , 
    Docx2txtLoader,
    TextLoader,
    
)
def load_document(file_path):

    extension = Path(file_path).suffix.lower()

    if extension == ".pdf":
        loader = PyPDFLoader(file_path)

    elif extension == ".docx":
        loader = Docx2txtLoader(file_path)

    elif extension in [".txt", ".md"]:
        loader = TextLoader(file_path, encoding="utf-8")

    else:
        raise ValueError(f"Unsupported file format: {extension}")

    documents = loader.load()

    return documents
