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
