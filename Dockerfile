FROM python:3.11-slim

WORKDIR /app

# System deps needed by lxml/bs4 parsing (harmless if unused).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# Render sets $PORT; bot.py reads it via os.environ.get("PORT", 8000)
CMD ["python", "bot.py"]
