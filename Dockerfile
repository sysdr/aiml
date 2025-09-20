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
