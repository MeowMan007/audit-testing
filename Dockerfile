FROM python:3.10-slim

# Install system dependencies for Selenium + Chrome
RUN apt-get update && apt-get install -y --no-install-recommends \
    chromium \
    chromium-driver \
    fonts-liberation \
    libnss3 \
    libxss1 \
    libasound2 \
    libatk-bridge2.0-0 \
    libgtk-3-0 \
    && rm -rf /var/lib/apt/lists/*

# Set Chrome/Chromedriver environment variables
ENV CHROME_BIN=/usr/bin/chromium
ENV CHROMEDRIVER_PATH=/usr/bin/chromedriver

# Create non-root user (HF Spaces requirement)
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Copy requirements first for Docker layer caching
COPY backend/requirements.txt ./backend/requirements.txt

# Install Python dependencies (CPU-only PyTorch to save space)
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    -r backend/requirements.txt

# Copy the rest of the application
COPY . .

# Create writable directories for the app user
RUN mkdir -p /app/data /app/dataset && \
    chown -R appuser:appuser /app

USER appuser

# HF Spaces expects port 7860
EXPOSE 7860

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
