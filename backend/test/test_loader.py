#1. Pdf loader
from backend.document_loader import load_document

documents = load_document("Deepak_Garg Resume.pdf")

for document in documents:
    print(document.page_content)
    print(document.metadata)

#2. PPT loader
from backend.document_loader import load_document

documents = load_document("Instagram_Project_Presentation.pptx")

for document in documents:
    print(document.page_content)
    print(document.metadata)
    
#3. CSV loader    
from backend.document_loader import load_document

documents = load_document("sample.csv")

for document in documents:
    print(document.page_content)
    print(document.metadata)
    
#4. xlsx loader
from backend.document_loader import load_document

documents = load_document("Python_AI_GenAI_6_Month_Internship_TOC.xlsx")

for document in documents:
    print(document.page_content)
    print(document.metadata)
    