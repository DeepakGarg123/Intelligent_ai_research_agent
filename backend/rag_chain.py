from backend.llm_provider import invoke_with_fallback
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()
prompt = ChatPromptTemplate.from_template(
    """Answer the question using only the provided context.

    If the answer is not available in the context,
    say that the document does not contain the answer.

    Context:
    {context}

    Question:
    {question}"""
)


def generate_answer(question, context):
    formatted_prompt = prompt.format(context=context, question=question)

    response = invoke_with_fallback(formatted_prompt)

    content = response.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []

        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))

            elif isinstance(block, str):
                text_parts.append(block)

        return "\n".join(text_parts)

    return str(content)
