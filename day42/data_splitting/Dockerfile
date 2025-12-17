FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY lesson_code.py test_lesson.py ./

CMD ["python", "lesson_code.py"]
