import google.generativeai as genai
from config.settings import settings
from app.models.chat import ChatMessage
from typing import List
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIService:
    def __init__(self):
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemini-3-flash-preview')
    
    def generate_response(self, user_message: str, conversation_history: List[ChatMessage] = []) -> str:
        try:
            # Build context from conversation history
            context = self._build_context(conversation_history)
            
            # Create full prompt with context
            full_prompt = f"{context}\nUser: {user_message}\nAssistant:"
            
            logger.info(f"Generating AI response for: {user_message[:50]}...")
            
            # Generate response using Gemini
            response = self.model.generate_content(full_prompt)
            
            if response and response.text:
                logger.info("AI response generated successfully")
                return response.text.strip()
            else:
                logger.error("Empty response from AI model")
                return "I apologize, but I couldn't generate a response at the moment."
                
        except Exception as e:
            logger.error(f"Error generating AI response: {str(e)}")
            return f"I'm experiencing technical difficulties: {str(e)}"
    
    def _build_context(self, conversation_history: List[ChatMessage]) -> str:
        if not conversation_history:
            return "You are a helpful AI assistant. Provide clear, concise, and helpful responses."
        
        context = "You are a helpful AI assistant. Here's our conversation so far:\n"
        for message in conversation_history[-5:]:  # Keep last 5 messages for context
            context += f"{message.role.title()}: {message.content}\n"
        
        return context

# Global AI service instance
ai_service = AIService()
