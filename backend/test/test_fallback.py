from unittest.mock import patch
from backend.llm_provider import invoke_with_fallback


with patch(
    "backend.llm_provider.get_gemini_model",
    side_effect=Exception("Simulated Gemini failure")
):

    with patch(
        "backend.llm_provider.get_groq_model",
        side_effect=Exception("Simulated Groq failure")
    ):

        response = invoke_with_fallback(
            "What is RAG? Explain in two lines."
        )

        print("\nFinal Response:")
        print(response.content)