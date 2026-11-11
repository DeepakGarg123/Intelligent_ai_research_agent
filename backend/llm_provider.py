from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
load_dotenv()
def get_gemini_model():
    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite"
    )


def get_ollama_model():
    return ChatOllama(
        model="llama3.2:latest"
    )

def invoke_with_fallback(prompt):
    try:
        gemini = get_gemini_model()
        response = gemini.invoke(prompt)
        return response

    except Exception as error:
        print("Gemini failed:", error)
        print("Switching to Ollama...")

        ollama = get_ollama_model()
        response = ollama.invoke(prompt)

        return response

if __name__ == "__main__":
    response = invoke_with_fallback(
        "Explain RAG in simple words."
    )

    print(response.content)