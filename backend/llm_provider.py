from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
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
    
def get_groq_model():
    return ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0
    )


def invoke_with_fallback(prompt):

    try:
        print("Using Gemini...")
        gemini = get_gemini_model()
        return gemini.invoke(prompt)

    except Exception as error:
        print("Gemini failed:", error)

    try:
        print("Switching to Groq...")
        groq = get_groq_model()
        return groq.invoke(prompt)

    except Exception as error:
        print("Groq failed:", error)

    try:
        print("Switching to Ollama...")
        ollama = get_ollama_model()
        return ollama.invoke(prompt)

    except Exception as error:
        print("Ollama failed:", error)
        raise RuntimeError("All LLM providers failed.")


if __name__ == "__main__":
    response = invoke_with_fallback(
        "Explain RAG in simple words."
    )

    print(response.content)
    