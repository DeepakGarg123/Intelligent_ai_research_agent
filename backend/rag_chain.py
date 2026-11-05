from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

def generate_answer(question , context):
    llm = ChatGoogleGenerativeAI(
        model  ='gemini-3.6-flash',
        temperature = 0
    )
    
    prompt = ChatPromptTemplate.from_template(
        """Answer the question using only the provided context.
        If the answer is not available in the context,
        say that document does not contain the answer.
        
        Context:{context}
        Question:{question}"""
    )
    
    chain = prompt|llm
    
    response = chain.invoke({
        "context":context,
        "question":question
    })
    return response.content
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    