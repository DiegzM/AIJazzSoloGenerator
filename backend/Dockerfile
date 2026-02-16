# Base image
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Document port
EXPOSE 5000

# Run gunicorn
CMD ["gunicorn", "--workers", "1", "--bind", "0.0.0.0:5000", "api.app:app", "--timeout", "120"]