from fastapi import APIRouter, HTTPException
from backend.models.request_models import ChatMessageRequest
from backend.models.response_models import ChatResponse
from backend.services.chat_service import get_chat_response
from backend.utils.vectorstore import get_vectorstore
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/", response_model=ChatResponse)
async def chat_with_resume(request: ChatMessageRequest):
    logger.info(f"Received chat request: '{request.message[:50]}...'")
    
    vectorstore = get_vectorstore()
    
    if vectorstore is None:
        logger.warning("Rejected chat query: FAISS Vectorstore not found. Resume was not uploaded properly.")
        raise HTTPException(status_code=400, detail="Upload resume first")
        
    try:
        logger.info("Vectorstore found, executing LLM RAG chat retrieval.")
        reply = get_chat_response(request.message)
        logger.info("Successfully generated AI response.")
        return ChatResponse(reply=reply)
    except Exception as e:
        logger.error(f"Error executing AI chat handler: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
