from pathlib import Path

import pymupdf

from langchain_core.documents import Document

from langchain_community.document_loaders import (
    Docx2txtLoader,
    TextLoader,
    UnstructuredPowerPointLoader,
    CSVLoader,
    UnstructuredExcelLoader
)

from backend.vision.vision_provider import extract_text_from_image

MIN_TEXT_CHARACTERS = 100


def load_pdf_with_vision(file_path):

    documents = []

    pdf = pymupdf.open(file_path)

    try:
        for page_number, page in enumerate(pdf):

            extracted_text = page.get_text("text").strip()

            if len(extracted_text) < MIN_TEXT_CHARACTERS:

                print(
                    f"Page {page_number + 1}: "
                    "Insufficient text. Trying Vision..."
                )

                pixmap = page.get_pixmap(
                    matrix=pymupdf.Matrix(2, 2),
                    alpha=False
                )

                image_bytes = pixmap.tobytes("png")

                try:
                    vision_text = extract_text_from_image(
                        image_bytes,
                        mime_type="image/png"
                    )

                    print(
                        f"Page {page_number + 1}: "
                        "Vision extraction successful."
                    )

                    if extracted_text:
                        extracted_text = (
                            vision_text
                            + "\n\nAdditional PDF text layer:\n"
                            + extracted_text
                        )
                    else:
                        extracted_text = vision_text

                except Exception as error:

                    if not extracted_text:
                        raise RuntimeError(
                            f"Vision extraction failed on "
                            f"page {page_number + 1}: {error}"
                        ) from error

                    print(
                        f"Vision failed on page {page_number + 1}: "
                        f"{error}"
                    )

            if extracted_text:

                documents.append(
                    Document(
                        page_content=extracted_text,
                        metadata={
                            "source": file_path.name,
                            "page": page_number
                        }
                    )
                )

    finally:
        pdf.close()

    return documents


def load_document(file_path):

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    if not file_path.is_file():
        raise ValueError(
            f"Not a valid file: {file_path}"
        )

    extension = file_path.suffix.lower()

    if extension == ".pdf":
        return load_pdf_with_vision(file_path)

    file_path = str(file_path)

    if extension == ".docx":
        loader = Docx2txtLoader(file_path)

    elif extension in [".txt", ".md"]:
        loader = TextLoader(file_path, encoding="utf-8")

    elif extension == ".pptx":
        loader = UnstructuredPowerPointLoader(file_path)

    elif extension == ".csv":
        loader = CSVLoader(file_path)

    elif extension == ".xlsx":
        loader = UnstructuredExcelLoader(
            file_path,
            mode="elements"
        )

    else:
        raise ValueError(
            f"Unsupported file format: {extension}"
        )

    documents = loader.load()

    return documents
