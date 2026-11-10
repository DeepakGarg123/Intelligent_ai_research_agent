
from backend.web_search import search_web

results = search_web("What is Retrieval Augmented Generation?")

for result in results:
    print("Title:", result["title"])
    print("URL:", result["url"])
    print("Content:", result["content"])
    print("-" * 50)
