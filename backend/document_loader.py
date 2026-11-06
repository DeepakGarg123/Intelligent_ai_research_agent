
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredPowerPointLoader,
    CSVLoader,
    UnstructuredExcelLoader
)


def load_document(file_path):
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not file_path.is_file():
        raise ValueError(f"Not a valid file: {file_path}")

    extension = file_path.suffix.lower()
    file_path = str(file_path)

    if extension == ".pdf":
        loader = PyPDFLoader(file_path)

    elif extension == ".docx":
        loader = Docx2txtLoader(file_path)

    elif extension in [".txt", ".md"]:
        loader = TextLoader(file_path, encoding="utf-8")

    elif extension == ".pptx":
        loader = UnstructuredPowerPointLoader(file_path)

    elif extension == ".csv":
        loader = CSVLoader(file_path)

    elif extension == ".xlsx":
        loader = UnstructuredExcelLoader(file_path, mode="elements")

    else:
        raise ValueError(f"Unsupported file format: {extension}")

    documents = loader.load()

    return documents
