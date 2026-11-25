
from pathlib import Path

from backend.document_loader import load_document


project_root = Path(__file__).resolve().parents[2]

pdf_path = project_root / "table_test.pdf"

documents = load_document(pdf_path)

print("\nPDF Extraction Results")
print("======================")

for document in documents:

    print("\nPage:", document.metadata["page"] + 1)
    print("Source:", document.metadata["source"])
    print("Extracted Text:")
    print("Total characters:" , len(document.page_content))
    print(document.page_content)
    print("----------------------")

print("\nTotal Pages Processed:", len(documents))
