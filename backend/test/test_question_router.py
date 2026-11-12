
from backend.question_router import route_question

questions = [
    "What is the objective of The Profitable Car Showroom project?",
    "List all projects covered in the data analytics course.",
    "What is RAG?",
    "Explain every project and its objective."
]

for question in questions:

    route = route_question(question)

    print("\nQuestion:", question)
    print("Selected Route:", route)
