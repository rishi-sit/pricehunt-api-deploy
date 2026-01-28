FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for Playwright
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY api_requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r api_requirements.txt

# Install Playwright browsers
RUN playwright install chromium && playwright install-deps chromium

# Copy application code
COPY . .

# Expose port (Koyeb will set PORT env var)
EXPOSE 8000

# Run the application
CMD ["sh", "-c", "uvicorn api_server:app --host 0.0.0.0 --port ${PORT:-8000}"]
