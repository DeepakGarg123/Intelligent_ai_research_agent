import json
from backend.llm_provider import invoke_with_fallback

def route_question(question):
    prompt = f"""
    You are a question classifier for a Hybrid RAG Document Assistant.

    Your task is to decide which retrieval strategy should be used
    to answer the user's question.

    There are only two possible routes:

    1. hybrid
       Use this for specific questions that need relevant information
       from a document.

       Examples:
       - What is RAG?
       - What is the objective of the Car Showroom project?
       - Explain the concept of machine learning.

    2. full_document
       Use this when the user wants a complete list, all items,
       every project, or comprehensive information across a document.

       Examples:
       - List all projects in the course.
       - What are all the projects and their objectives?
       - List every topic covered in this document.
       - Give me all the projects mentioned in the PDF.

    Important rules:
    - Choose only one route.
    - Return only valid JSON.
    - Do not explain your decision.

    User question: {question}

    Return format:
    {{
        "route": "hybrid"
    }}

    or

    {{
        "route": "full_document"
    }}
    """

    response = invoke_with_fallback(prompt)

    content = response.content

    if isinstance(content, list):
        content = "\n".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )

    try:
        result = json.loads(content)
        route = result.get("route")

        if route in ["hybrid", "full_document"]:
            return route

    except (json.JSONDecodeError, AttributeError, TypeError):
        pass

    return "hybrid"
