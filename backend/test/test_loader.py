from backend.document_loader import load_document
file_path = "Deepak_Garg Resume.pdf"
documents = load_document(file_path)

print("Total document sections:",len(documents))
print(documents[0].page_content)
print(documents[0].metadata)