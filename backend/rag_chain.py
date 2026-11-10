
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()


def generate_answer(question, context):

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        temperature=0
    )

    prompt = ChatPromptTemplate.from_template(
        """Answer the question using only the provided context.

        If the answer is not available in the context,
        say that the document does not contain the answer.

        Context:
        {context}

        Question:
        {question}"""
    )

    chain = prompt | llm

    response = chain.invoke({
        "context": context,
        "question": question
    })

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
