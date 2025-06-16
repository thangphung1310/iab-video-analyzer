# Use Python with Debian for faster builds (pre-compiled wheels available)
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p uploads uploads/models uploads/models/whisper_cache && \
    chmod -R 755 uploads

RUN groupadd -g 1001 pythonuser && \
    useradd -r -u 1001 -g pythonuser pythonuser
RUN chown -R pythonuser:pythonuser /app
USER pythonuser

EXPOSE 3001

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:3001/api/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "3001"] 
