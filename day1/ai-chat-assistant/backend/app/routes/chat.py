from fastapi import APIRouter, HTTPException
from app.models.chat import ChatRequest, ChatResponse, ChatMessage
from app.services.ai_service import ai_service
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        # Validate input
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        logger.info(f"Processing chat request: {request.message[:50]}...")
        
        # Generate AI response
        ai_response = ai_service.generate_response(
            request.message, 
            request.conversation_history or []
        )
        
        return ChatResponse(
            response=ai_response,
            success=True,
            timestamp=datetime.now()
        )
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {str(e)}")
        return ChatResponse(
            response="I apologize for the technical difficulties. Please try again.",
            success=False,
            error_message=str(e),
            timestamp=datetime.now()
        )

@router.get("/health")
async def health_check():
    return {"status": "healthy", "service": "AI Chat Assistant"}
