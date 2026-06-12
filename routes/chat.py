from fastapi import APIRouter, HTTPException
from models.request_models import ChatMessageRequest
from models.response_models import ChatResponse
from services.chat_service import get_chat_response
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/", response_model=ChatResponse)
async def chat_with_resume(request: ChatMessageRequest):
    logger.info(f"Chat request (session={request.session_id}): '{request.message[:50]}...'")
    # No hard 400: works without a resume (general mode), uses RAG when ready.
    try:
        reply = get_chat_response(request.message, request.session_id)
        return ChatResponse(reply=reply)
    except Exception as e:
        logger.error(f"Error in chat handler: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
