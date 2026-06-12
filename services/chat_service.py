import os
import logging
from dotenv import load_dotenv

from utils.vectorstore import get_vectorstore, get_skills, get_roles

load_dotenv()
logger = logging.getLogger(__name__)


PERSONA = (
    "You are HireGenei, a sharp, encouraging AI career consultant. "
    "Be concrete and practical. Prefer short, scannable bullet points over long paragraphs. "
    "When relevant, end with 2-3 concrete next steps the user can act on today. "
    "If the user's question is vague, ask one short clarifying question before advising. "
    "Be honest about trade-offs; never invent job listings, salaries, or company facts."
)

RAG_TEMPLATE = (
    PERSONA
    + "\n\nThe user has this resume-derived profile:\n"
    "Skills: {skills}\nMatched roles: {roles}\n\n"
    "Use the resume context below when it helps answer.\n"
    "Question: {input}\n"
    "Resume context: {context}\n"
)

GENERAL_TEMPLATE = (
    PERSONA
    + "\n\n(No resume has been uploaded yet, so answer from general career knowledge. "
    "If a personalized answer would need their resume, gently mention they can upload one.)\n\n"
    "Known skills (may be empty): {skills}\n"
    "Known roles (may be empty): {roles}\n\n"
    "Question: {input}\n"
)


def _get_llm():
    from langchain_groq import ChatGroq
    from core.config import settings

    if not settings.GROQ_API_KEY:
        raise EnvironmentError(
            "GROQ_API_KEY is not configured for the backend. "
            "Set GROQ_API_KEY in backend/.env or in the OS environment before starting the service."
        )
    return ChatGroq(api_key=settings.GROQ_API_KEY, model_name="llama-3.1-8b-instant")


def get_chat_response(message: str, session_id: str | None = None) -> str:
    # Lazy imports keep app startup fast and avoid Render port-detection timeouts.
    from langchain_core.prompts import ChatPromptTemplate

    llm = _get_llm()

    vectorstore = get_vectorstore(session_id)
    skills = ", ".join(get_skills(session_id)) or "none yet"
    roles = ", ".join(get_roles(session_id)) or "none yet"

    import time
    start_time = time.time()

    # ---- General mode: no resume / vectorstore not ready yet ----
    if vectorstore is None:
        logger.info("Chat in GENERAL mode (no vectorstore for this session).")
        prompt = ChatPromptTemplate.from_template(GENERAL_TEMPLATE)
        chain = prompt | llm
        resp = chain.invoke({"input": message, "skills": skills, "roles": roles})
        answer = getattr(resp, "content", str(resp))
        logger.info(f"General chat done in {time.time() - start_time:.2f}s.")
        return answer or "I couldn't generate a response. Please try rephrasing."

    # ---- RAG mode: resume vectorstore available ----
    try:
        from langchain.chains.combine_documents import create_stuff_documents_chain
        from langchain.chains import create_retrieval_chain
    except Exception:
        from langchain_classic.chains.combine_documents import create_stuff_documents_chain
        from langchain_classic.chains import create_retrieval_chain

    logger.info("Chat in RAG mode (resume vectorstore found).")
    prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
    retriever = vectorstore.as_retriever()
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    try:
        response = retrieval_chain.invoke({"input": message, "skills": skills, "roles": roles})
        answer = response.get("answer", "No relevant answer found.")
        logger.info(f"RAG chat done in {time.time() - start_time:.2f}s. Answer length: {len(answer)}")
        return answer
    except Exception as e:
        logger.error(f"Error during Groq chat retrieval: {str(e)}", exc_info=True)
        raise e
