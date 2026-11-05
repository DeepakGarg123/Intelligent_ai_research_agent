from backend.document_loader import load_document
from backend.chunker import split_documents

documents = load_document("Deepak_Garg Resume.pdf")
chunks = split_documents(documents)

print("Total chunks:" , len(chunks))

print("First Chunk:" , chunks[0].page_content)

print("Chunk metadata:" , chunks[0].metadata)

