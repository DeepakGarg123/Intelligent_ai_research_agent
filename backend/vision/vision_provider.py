
import os
import base64

from dotenv import load_dotenv
from google import genai
from google.genai import types
from groq import Groq

load_dotenv()

GEMINI_VISION_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.5-flash-lite"
]

GROQ_VISION_MODEL = "qwen/qwen3.8-27b"


EXTRACTION_PROMPT = """
Extract all readable information from this document image.

Follow these instructions carefully:

1. Preserve the original wording, numbers, dates and units.
2. Preserve the original document's reading order.
3. For tables:
   - Convert tables into Markdown table format.
   - Preserve all column headers and row labels.
   - Keep every value under its correct column.
   - Do not merge unrelated rows or columns.
4. For charts and graphs:
   - Extract the chart title.
   - Identify the X-axis and Y-axis labels.
   - Extract category names and their corresponding values.
   - Preserve units and legends.
   - Clearly represent each category-value relationship.
5. For diagrams:
   - Describe the visible components and their relationships.
   - Preserve labels, arrows and connections wherever readable.
6. For checkboxes:
   - Checked checkbox: [✓]
   - Unchecked checkbox: [ ]
7. Preserve handwritten text wherever readable.
8. Mark unclear text or values as [UNREADABLE].
9. Never guess missing information or calculate values that are not shown.
10. Do not summarize or interpret the document.
11. Keep paragraphs, headings and sections in their original order.

Return the extracted information in clean Markdown format.
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
    print("Sending Image to Gemini")

    response = client.models.generate_content(
        model=model_name,
        contents=[
            EXTRACTION_PROMPT,
            image_part
        ]
        
    )
    print("response received")

    if not response.text:
        raise ValueError("Gemini returned an empty response.")

    return response.text


def extract_with_groq(image_bytes, mime_type):

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY is missing.")

    client = Groq(api_key=api_key)

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
