FROM python:3.11-slim

# System deps (optional: libpq already bundled via psycopg[binary])
RUN pip install --no-cache-dir --upgrade pip

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py ./main.py
COPY templates ./templates

EXPOSE 8000
