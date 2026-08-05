FROM python:3.11-slim

# ffmpeg powers the v3 multiagent pipeline's frame + audio extraction.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY bot.py .
COPY vexi/ ./vexi/

CMD ["python", "bot.py"]
