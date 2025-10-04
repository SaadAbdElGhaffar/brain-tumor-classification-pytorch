FROM python:3.11-slim

WORKDIR /app

# Copy requirements from root directory
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy app
COPY api/app.py .

# Expose port
EXPOSE 8000

# Run app
CMD ["python", "app.py"]