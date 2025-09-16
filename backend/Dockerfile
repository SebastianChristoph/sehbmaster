FROM python:3.11-slim

WORKDIR /app

# System deps (psycopg braucht libpq)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
COPY alembic.ini .

RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8000
