import os
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from utils.vectorstore import get_vectorstore, get_skills, get_roles
import logging

logger = logging.getLogger(__name__)

prompt = ChatPromptTemplate.from_template(
    """You are an intelligent job consultant helping the user.
The user has the following resume-based skills and job roles:
Skills: {skills}
Roles: {roles}

Always answer in short bullet points with clarity and avoid long paragraphs.

Question: {input}
Context (Scraped from Resume Documents): {context}
"""
)

def get_chat_response(message: str) -> str:
    vectorstore = get_vectorstore()
    
    # We shouldn't hit this if the router validates, but just in case
    if vectorstore is None:
        raise ValueError("Vectorstore not initialized")

    retriever = vectorstore.as_retriever()
    
    from core.config import settings

    if not settings.GROQ_API_KEY:
        raise EnvironmentError(
            "GROQ_API_KEY is not configured for the backend. "
            "Set GROQ_API_KEY in backend/.env or in the OS environment before starting the service."
        )

    llm = ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model_name="llama-3.1-8b-instant"
    )

    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Inject global scope context
    user_skills_raw = get_skills()
    user_roles_raw = get_roles()
    
    context_vars = {
        "input": message,
        "skills": ", ".join(user_skills_raw),
        "roles": ", ".join(user_roles_raw),
    }

    import time
    start_time = time.time()
    try:
        logger.info(f"Invoking Groq API for LLaMA-3 with RAG Context. Query: '{message[:30]}'")
        response = retrieval_chain.invoke(context_vars)
        elapsed = time.time() - start_time
        
        answer = response.get("answer", "⚠️ No relevant answer found.")
        logger.info(f"Groq API executed successfully in {elapsed:.2f}s. Answer length: {len(answer)}")
        return answer
    except Exception as e:
        logger.error(f"Error during Groq chat retrieval: {str(e)}", exc_info=True)
        raise e
