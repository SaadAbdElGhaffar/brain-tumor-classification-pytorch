FROM python:3.11-slim

WORKDIR /app

# Copy requirements from root directory
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy app
COPY streamlit/app.py .

# Expose port
EXPOSE 8501

# Run app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]