
from backend.document_loader import load_document
from backend.chunker import split_documents
from backend.full_document_search import create_batches
from backend.full_document_search import extract_information_from_batch

documents = load_document("Codewithharry.pdf")

chunks = split_documents(documents)

batches = create_batches(chunks, batch_size=20)

print("Total chunks:", len(chunks))
print("Total batches:", len(batches))

for index, batch in enumerate(batches):
    print(f"Batch {index + 1}: {len(batch)} chunks")


from backend.full_document_search import search_entire_document

query = input("What do you want to find? ")

results = search_entire_document(chunks, query)

print("\nExtracted Information:\n")

for item in results:
    print(item)
    print("-" * 60)

print("Total items found:", len(results))

    