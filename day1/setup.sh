#!/bin/bash

# Day 1: Python Basics for AI Systems - Complete Implementation Script
# Creates a fully functional AI chat assistant with Python backend and React frontend

set -e  # Exit on any error

echo "🚀 Starting Day 1: Python Basics for AI Systems Implementation"

# Create project structure
echo "📁 Creating project structure..."
mkdir -p ai-chat-assistant/{backend,frontend,tests,docs,scripts}
cd ai-chat-assistant

# Create backend directory structure
mkdir -p backend/{app,tests,config}
mkdir -p backend/app/{routes,services,models}

# Create frontend directory structure  
mkdir -p frontend/{src,public,tests}
mkdir -p frontend/src/{components,services,styles,utils}

echo "🐍 Setting up Python backend..."

# Create Python requirements.txt
cat > backend/requirements.txt << 'EOF'
fastapi==0.104.1
uvicorn==0.24.0
python-dotenv==1.0.0
google-generativeai==0.3.2
pydantic==2.5.0
pytest==7.4.3
httpx==0.25.2
requests==2.31.0
python-multipart==0.0.6
jinja2==3.1.2
EOF

# Create backend configuration and source files first (so all files exist even if pip fails)
cd backend
cat > config/settings.py << 'EOF'
import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    GEMINI_API_KEY = "AIzaSyDGswqDT4wQw_bd4WZtIgYAawRDZ0Gisn8"
    APP_NAME = "AI Chat Assistant"
    DEBUG = True
    HOST = "0.0.0.0"
    PORT = 8000
    CORS_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]

settings = Settings()
EOF

# Create Pydantic models
cat > app/models/chat.py << 'EOF'
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = None

class ChatRequest(BaseModel):
    message: str
    conversation_history: Optional[List[ChatMessage]] = []

class ChatResponse(BaseModel):
    response: str
    success: bool
    error_message: Optional[str] = None
    timestamp: datetime = None
EOF

# Create AI service
cat > app/services/ai_service.py << 'EOF'
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
        self.model = genai.GenerativeModel('gemini-pro')
    
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
EOF

# Create API routes
cat > app/routes/chat.py << 'EOF'
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
EOF

# Create main FastAPI application
cat > app/main.py << 'EOF'
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes.chat import router as chat_router
from config.settings import settings

app = FastAPI(
    title=settings.APP_NAME,
    description="AI Chat Assistant powered by Gemini AI",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat_router, prefix="/api/v1", tags=["chat"])

@app.get("/")
async def root():
    return {"message": "AI Chat Assistant API is running!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.HOST, port=settings.PORT, debug=settings.DEBUG)
EOF

# Create backend tests
mkdir -p tests
cat > tests/test_chat.py << 'EOF'
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_chat_endpoint():
    response = client.post(
        "/api/v1/chat",
        json={
            "message": "Hello, how are you?",
            "conversation_history": []
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] == True
    assert len(data["response"]) > 0

def test_empty_message():
    response = client.post(
        "/api/v1/chat",
        json={
            "message": "",
            "conversation_history": []
        }
    )
    assert response.status_code == 400
EOF

# Create Python virtual environment and install dependencies
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt || true  # Continue so frontend and scripts are still created if pip fails

cd ..  # Back to project root

echo "⚛️ Setting up React frontend..."

# Create package.json for React frontend
cat > frontend/package.json << 'EOF'
{
  "name": "ai-chat-frontend",
  "version": "1.0.0",
  "private": true,
  "dependencies": {
    "@testing-library/jest-dom": "^6.1.5",
    "@testing-library/react": "^13.4.0",
    "@testing-library/user-event": "^14.5.1",
    "axios": "^1.6.2",
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-scripts": "5.0.1",
    "styled-components": "^6.1.1",
    "lucide-react": "^0.294.0"
  },
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build",
    "test": "react-scripts test",
    "eject": "react-scripts eject"
  },
  "eslintConfig": {
    "extends": [
      "react-app",
      "react-app/jest"
    ]
  },
  "browserslist": {
    "production": [
      ">0.2%",
      "not dead",
      "not op_mini all"
    ],
    "development": [
      "last 1 chrome version",
      "last 1 firefox version",
      "last 1 safari version"
    ]
  },
  "proxy": "http://localhost:8000"
}
EOF

# Create public/index.html
cat > frontend/public/index.html << 'EOF'
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <link rel="icon" href="%PUBLIC_URL%/favicon.ico" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="theme-color" content="#000000" />
    <meta name="description" content="AI Chat Assistant - Learn Python and AI" />
    <title>AI Chat Assistant</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  </head>
  <body>
    <noscript>You need to enable JavaScript to run this app.</noscript>
    <div id="root"></div>
  </body>
</html>
EOF

# Create API service
cat > frontend/src/services/api.js << 'EOF'
import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api/v1';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const chatAPI = {
  sendMessage: async (message, conversationHistory = []) => {
    try {
      const response = await apiClient.post('/chat', {
        message,
        conversation_history: conversationHistory,
      });
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.detail || 'Failed to send message');
    }
  },

  healthCheck: async () => {
    try {
      const response = await apiClient.get('/health');
      return response.data;
    } catch (error) {
      throw new Error('Health check failed');
    }
  },
};

export default apiClient;
EOF

# Create styled components
cat > frontend/src/styles/GlobalStyles.js << 'EOF'
import { createGlobalStyle } from 'styled-components';

export const GlobalStyles = createGlobalStyle`
  * {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
  }

  body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: #333;
    min-height: 100vh;
  }

  #root {
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 20px;
  }
`;

export const theme = {
  colors: {
    primary: '#4f46e5',
    primaryLight: '#6366f1',
    secondary: '#10b981',
    background: '#ffffff',
    surface: '#f8fafc',
    text: '#1f2937',
    textLight: '#6b7280',
    border: '#e5e7eb',
    error: '#ef4444',
    success: '#10b981',
    warning: '#f59e0b',
  },
  shadows: {
    sm: '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
    md: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
    lg: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
    xl: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
  },
  borderRadius: {
    sm: '0.25rem',
    md: '0.375rem',
    lg: '0.5rem',
    xl: '0.75rem',
    '2xl': '1rem',
  },
};
EOF

# Create Chat components
cat > frontend/src/components/ChatContainer.js << 'EOF'
import React, { useState, useEffect, useRef } from 'react';
import styled from 'styled-components';
import { Send, Bot, User, Loader, AlertCircle } from 'lucide-react';
import { chatAPI } from '../services/api';

const Container = styled.div`
  width: 100%;
  max-width: 800px;
  height: 600px;
  background: ${props => props.theme.colors.background};
  border-radius: ${props => props.theme.borderRadius['2xl']};
  box-shadow: ${props => props.theme.shadows.xl};
  display: flex;
  flex-direction: column;
  overflow: hidden;
`;

const Header = styled.div`
  background: ${props => props.theme.colors.primary};
  color: white;
  padding: 1.5rem;
  text-align: center;
`;

const Title = styled.h1`
  font-size: 1.5rem;
  font-weight: 600;
  margin: 0;
`;

const Subtitle = styled.p`
  font-size: 0.875rem;
  opacity: 0.9;
  margin: 0.5rem 0 0 0;
`;

const MessagesContainer = styled.div`
  flex: 1;
  overflow-y: auto;
  padding: 1rem;
  display: flex;
  flex-direction: column;
  gap: 1rem;
`;

const Message = styled.div`
  display: flex;
  align-items: flex-start;
  gap: 0.75rem;
  ${props => props.$isUser && 'flex-direction: row-reverse;'}
`;

const MessageIcon = styled.div`
  width: 2rem;
  height: 2rem;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  background: ${props => props.$isUser ? props.theme.colors.primary : props.theme.colors.secondary};
  color: white;
`;

const MessageBubble = styled.div`
  max-width: 70%;
  padding: 0.75rem 1rem;
  border-radius: ${props => props.theme.borderRadius.lg};
  background: ${props => props.$isUser ? props.theme.colors.primary : props.theme.colors.surface};
  color: ${props => props.$isUser ? 'white' : props.theme.colors.text};
  word-wrap: break-word;
  line-height: 1.5;
`;

const InputContainer = styled.div`
  padding: 1rem;
  border-top: 1px solid ${props => props.theme.colors.border};
  background: ${props => props.theme.colors.surface};
`;

const InputWrapper = styled.div`
  display: flex;
  gap: 0.5rem;
  align-items: flex-end;
`;

const TextArea = styled.textarea`
  flex: 1;
  min-height: 2.5rem;
  max-height: 6rem;
  padding: 0.75rem;
  border: 2px solid ${props => props.theme.colors.border};
  border-radius: ${props => props.theme.borderRadius.lg};
  font-family: inherit;
  font-size: 0.875rem;
  resize: vertical;
  transition: border-color 0.2s;

  &:focus {
    outline: none;
    border-color: ${props => props.theme.colors.primary};
  }

  &:disabled {
    background: ${props => props.theme.colors.border};
    cursor: not-allowed;
  }
`;

const SendButton = styled.button`
  width: 2.5rem;
  height: 2.5rem;
  border: none;
  border-radius: ${props => props.theme.borderRadius.lg};
  background: ${props => props.theme.colors.primary};
  color: white;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s;

  &:hover:not(:disabled) {
    background: ${props => props.theme.colors.primaryLight};
    transform: translateY(-1px);
  }

  &:disabled {
    background: ${props => props.theme.colors.border};
    cursor: not-allowed;
    transform: none;
  }
`;

const LoadingMessage = styled.div`
  display: flex;
  align-items: center;
  gap: 0.5rem;
  color: ${props => props.theme.colors.textLight};
  font-style: italic;
`;

const ErrorMessage = styled.div`
  display: flex;
  align-items: center;
  gap: 0.5rem;
  color: ${props => props.theme.colors.error};
  background: ${props => props.theme.colors.error}10;
  padding: 0.75rem;
  border-radius: ${props => props.theme.borderRadius.md};
  margin: 0.5rem 0;
`;

const ChatContainer = () => {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: 'Hello! I\'m your AI assistant. I\'m here to help you learn Python and AI. What would you like to know?',
      timestamp: new Date().toISOString(),
    },
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    const el = messagesEndRef.current;
    if (el && typeof el.scrollIntoView === 'function') {
      el.scrollIntoView({ behavior: 'smooth' });
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return;

    const userMessage = {
      role: 'user',
      content: inputMessage.trim(),
      timestamp: new Date().toISOString(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);
    setError('');

    try {
      const response = await chatAPI.sendMessage(userMessage.content, messages);
      
      if (response.success) {
        const assistantMessage = {
          role: 'assistant',
          content: response.response,
          timestamp: new Date().toISOString(),
        };
        setMessages(prev => [...prev, assistantMessage]);
      } else {
        throw new Error(response.error_message || 'Failed to get response');
      }
    } catch (err) {
      setError(err.message);
      console.error('Chat error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <Container>
      <Header>
        <Title>AI Chat Assistant</Title>
        <Subtitle>Powered by Python & Gemini AI - Day 1 Project</Subtitle>
      </Header>

      <MessagesContainer>
        {messages.map((message, index) => (
          <Message key={index} $isUser={message.role === 'user'}>
            <MessageIcon $isUser={message.role === 'user'}>
              {message.role === 'user' ? <User size={16} /> : <Bot size={16} />}
            </MessageIcon>
            <MessageBubble $isUser={message.role === 'user'}>
              {message.content}
            </MessageBubble>
          </Message>
        ))}

        {isLoading && (
          <LoadingMessage>
            <Loader size={16} className="animate-spin" />
            AI is thinking...
          </LoadingMessage>
        )}

        {error && (
          <ErrorMessage>
            <AlertCircle size={16} />
            {error}
          </ErrorMessage>
        )}

        <div ref={messagesEndRef} />
      </MessagesContainer>

      <InputContainer>
        <InputWrapper>
          <TextArea
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Type your message here..."
            disabled={isLoading}
            rows={1}
          />
          <SendButton
            onClick={handleSendMessage}
            disabled={isLoading || !inputMessage.trim()}
          >
            {isLoading ? <Loader size={16} /> : <Send size={16} />}
          </SendButton>
        </InputWrapper>
      </InputContainer>
    </Container>
  );
};

export default ChatContainer;
EOF

# Create main App component
cat > frontend/src/App.js << 'EOF'
import React from 'react';
import { ThemeProvider } from 'styled-components';
import { GlobalStyles, theme } from './styles/GlobalStyles';
import ChatContainer from './components/ChatContainer';

function App() {
  return (
    <ThemeProvider theme={theme}>
      <GlobalStyles />
      <div className="App">
        <ChatContainer />
      </div>
    </ThemeProvider>
  );
}

export default App;
EOF

# Create index.js
cat > frontend/src/index.js << 'EOF'
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
EOF

# Create frontend tests
cat > frontend/src/App.test.js << 'EOF'
import { render, screen } from '@testing-library/react';
import App from './App';

test('renders AI Chat Assistant', () => {
  render(<App />);
  const titleElement = screen.getByText(/AI Chat Assistant/i);
  expect(titleElement).toBeInTheDocument();
});
EOF

# Create test setup (jest-dom matchers)
cat > frontend/src/setupTests.js << 'EOF'
import '@testing-library/jest-dom';
EOF

echo "🔧 Creating build and deployment scripts..."

# Create build.sh
cat > build.sh << 'EOF'
#!/bin/bash

echo "🚀 Building AI Chat Assistant..."

# Build Backend
echo "🐍 Building Python Backend..."
cd backend

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python3.11 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Run backend tests
echo "🧪 Running backend tests..."
python -m pytest tests/ -v

echo "✅ Backend build complete!"

cd ..

# Build Frontend
echo "⚛️ Building React Frontend..."
cd frontend

# Install Node.js dependencies
npm install

# Run frontend tests
echo "🧪 Running frontend tests..."
npm test -- --coverage --watchAll=false

# Build for production
echo "📦 Building for production..."
npm run build

echo "✅ Frontend build complete!"

cd ..

echo "🎉 Build completed successfully!"
echo "📋 Next steps:"
echo "   1. Run './start.sh' to start the application"
echo "   2. Open http://localhost:3000 in your browser"
echo "   3. Start chatting with your AI assistant!"
EOF

# Create start.sh
cat > start.sh << 'EOF'
#!/bin/bash

echo "🚀 Starting AI Chat Assistant..."

# Function to cleanup on exit
cleanup() {
    echo "🛑 Shutting down services..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start Backend
echo "🐍 Starting Python Backend..."
cd backend
source venv/bin/activate
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

cd ..

# Wait for backend to start
echo "⏳ Waiting for backend to start..."
sleep 5

# Start Frontend
echo "⚛️ Starting React Frontend..."
cd frontend
npm start &
FRONTEND_PID=$!

cd ..

echo "✅ Services started successfully!"
echo "🌐 Frontend: http://localhost:3000"
echo "📡 Backend API: http://localhost:8000"
echo "📖 API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for services
wait
EOF

# Create stop.sh
cat > stop.sh << 'EOF'
#!/bin/bash

echo "🛑 Stopping AI Chat Assistant..."

# Kill all Node.js processes (React)
pkill -f "react-scripts start" 2>/dev/null

# Kill all Python/uvicorn processes
pkill -f "uvicorn" 2>/dev/null

# Kill any remaining processes on ports 3000 and 8000
lsof -ti:3000 | xargs kill -9 2>/dev/null
lsof -ti:8000 | xargs kill -9 2>/dev/null

echo "✅ All services stopped!"
EOF

# Create demo.sh
cat > demo.sh << 'EOF'
#!/bin/bash

echo "🎬 Running AI Chat Assistant Demo..."

# Check if services are running
backend_status=$(curl -s http://localhost:8000/api/v1/health 2>/dev/null | grep -o '"status":"healthy"' || echo "")
frontend_status=$(curl -s http://localhost:3000 2>/dev/null && echo "running" || echo "")

if [ -z "$backend_status" ] || [ -z "$frontend_status" ]; then
    echo "⚠️  Services not running. Starting them first..."
    ./start.sh &
    sleep 10
fi

echo "🧪 Testing API endpoints..."

# Test health endpoint
echo "1. Testing health endpoint..."
curl -X GET http://localhost:8000/api/v1/health

echo -e "\n\n2. Testing chat endpoint..."
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello! Can you explain what Python is?",
    "conversation_history": []
  }'

echo -e "\n\n✅ Demo complete!"
echo "🌐 Open http://localhost:3000 to interact with the UI"
echo "📡 API Documentation: http://localhost:8000/docs"
EOF

# Make scripts executable
chmod +x build.sh start.sh stop.sh demo.sh

# Create Docker configuration
cat > Dockerfile << 'EOF'
# Multi-stage Dockerfile for AI Chat Assistant

# Backend Stage
FROM python:3.11-slim as backend

WORKDIR /app/backend

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Frontend Stage
FROM node:18-alpine as frontend

WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm ci --only=production

COPY frontend/ .
RUN npm run build

# Production Stage
FROM nginx:alpine

# Copy backend files
COPY --from=backend /app/backend /app/backend

# Copy frontend build
COPY --from=frontend /app/frontend/build /usr/share/nginx/html

# Copy nginx config
COPY nginx.conf /etc/nginx/nginx.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
EOF

# Create docker-compose.yml
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  backend:
    build:
      context: .
      target: backend
    ports:
      - "8000:8000"
    environment:
      - PYTHONPATH=/app/backend
    volumes:
      - ./backend:/app/backend
    networks:
      - ai-chat-network

  frontend:
    build:
      context: .
      target: frontend
    ports:
      - "3000:3000"
    environment:
      - REACT_APP_API_URL=http://localhost:8000/api/v1
    volumes:
      - ./frontend/src:/app/frontend/src
    networks:
      - ai-chat-network
    depends_on:
      - backend

networks:
  ai-chat-network:
    driver: bridge
EOF

# Create nginx config
cat > nginx.conf << 'EOF'
events {
    worker_connections 1024;
}

http {
    upstream backend {
        server backend:8000;
    }

    server {
        listen 80;
        
        location / {
            root /usr/share/nginx/html;
            index index.html index.htm;
            try_files $uri $uri/ /index.html;
        }

        location /api/ {
            proxy_pass http://backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }
    }
}
EOF

# Create Docker build script
cat > docker-build.sh << 'EOF'
#!/bin/bash

echo "🐳 Building Docker containers..."

# Build and start with Docker Compose
docker-compose up --build -d

echo "✅ Docker containers started!"
echo "🌐 Application: http://localhost:3000"
echo "📡 API: http://localhost:8000"

# Show logs
docker-compose logs -f
EOF

chmod +x docker-build.sh

# Create .gitignore
cat > .gitignore << 'EOF'
# Python
backend/venv/
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
*.egg-info/
.eggs/
dist/
build/
.pytest_cache/
.coverage
htmlcov/
.env
.env.local
*.log

# Node
frontend/node_modules/
frontend/build/
frontend/.env
frontend/.env.local
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# IDE
.idea/
.vscode/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
EOF

# Create README
cat > README.md << 'EOF'
# Day 1: AI Chat Assistant - Python Basics for AI Systems

A full-stack AI chat application demonstrating Python fundamentals in AI system architecture.

## 🚀 Quick Start

### Option 1: Local Development
```bash
# Build the project
./build.sh

# Start all services
./start.sh

# Run demo tests
./demo.sh
```

### Option 2: Docker
```bash
# Build and start with Docker
./docker-build.sh
```

## 🌐 Access Points

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## 🛠 Tech Stack

- **Backend**: Python 3.11, FastAPI, Uvicorn
- **Frontend**: React 18, Styled Components
- **AI**: Google Gemini AI
- **Testing**: Pytest, Jest
- **Deployment**: Docker, Docker Compose

## 📁 Project Structure

```
ai-chat-assistant/
├── backend/                 # Python FastAPI backend
│   ├── app/
│   │   ├── models/         # Pydantic models
│   │   ├── routes/         # API endpoints
│   │   └── services/       # Business logic
│   ├── config/             # Configuration
│   └── tests/              # Backend tests
├── frontend/               # React frontend
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── services/       # API client
│   │   └── styles/         # Styled components
│   └── public/             # Static files
└── scripts/                # Build/deployment scripts
```

## 🧪 Testing

```bash
# Backend tests
cd backend && python -m pytest tests/ -v

# Frontend tests  
cd frontend && npm test
```

## 🔧 Development

```bash
# Start backend only
cd backend && source venv/bin/activate && python -m uvicorn app.main:app --reload

# Start frontend only
cd frontend && npm start
```

## 📚 Learning Outcomes

- Python virtual environments and dependency management
- FastAPI for building RESTful APIs
- React for modern frontend development
- AI API integration with error handling
- Full-stack application architecture
- Testing strategies for AI applications
EOF

echo "✅ Implementation complete!"
echo ""
echo "🎯 Next Steps:"
echo "1. Run './build.sh' to build the project"
echo "2. Run './start.sh' to start all services"
echo "3. Open http://localhost:3000 in your browser"
echo "4. Start chatting with your AI assistant!"
echo ""
echo "🐳 Docker Alternative:"
echo "1. Run './docker-build.sh' for containerized deployment"
echo ""
echo "📖 Check README.md for detailed instructions"