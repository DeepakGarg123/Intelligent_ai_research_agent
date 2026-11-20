import os
from google import genai
from dotenv import load_dotenv
from google.genai import types
from groq import Groq

load_dotenv
GEMINI_VISION_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash"
]

GROQ_VISION_MODEL = "qwen/qwen3.8-27b"

EXTRACTION_PROMPT = """
Extract all readable text from this document image.

Instructions:
1. Preserve the original wording and numbers.
2. Preserve tables and their column relationships.
3. Do not summarize or interpret the content.
4. Do not guess unreadable words or numbers.
5. Mark unclear text as [UNREADABLE].
6. Return only extracted text.
"""

def extract_with_gemini(image_bytes, mime_type, model_name):

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing.")

    client = genai.Client(api_key=api_key)

    image_part = types.Part.from_bytes(
        data=image_bytes,
        mime_type=mime_type
    )

    response = client.models.generate_content(
        model=model_name,
        contents=[
            EXTRACTION_PROMPT,
            image_part
        ]
    )

    if not response.text:
        raise ValueError("Gemini returned an empty response.")

    return response.text


def extract_with_groq(image_bytes, mime_type):

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY is missing.")

    client = Groq(api_key=api_key)

    import base64

    encoded_image = base64.b64encode(image_bytes).decode("utf-8")

    response = client.chat.completions.create(
        model=GROQ_VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": EXTRACTION_PROMPT
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{encoded_image}"
                        }
                    }
                ]
            }
        ],
        max_completion_tokens=4096
    )

    content = response.choices[0].message.content

    if not content:
        raise ValueError("Groq returned an empty response.")

    return content


def extract_text_from_image(image_bytes, mime_type="image/png"):

    for model_name in GEMINI_VISION_MODELS:

        try:
            print(f"Trying Gemini Vision: {model_name}")

            result = extract_with_gemini(
                image_bytes,
                mime_type,
                model_name
            )

            print(f"Success: {model_name}")
            return result

        except Exception as error:
            print(f"{model_name} failed: {error}")

    try:
        print("Switching to Groq Vision...")

        result = extract_with_groq(
            image_bytes,
            mime_type
        )

        print("Groq Vision succeeded.")
        return result

    except Exception as error:
        print(f"Groq Vision failed: {error}")

    raise RuntimeError("All vision models failed.")

