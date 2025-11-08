# streamlit/Dockerfile
FROM python:3.11-slim-bookworm

WORKDIR /app

# wkhtmltopdf + fonts so the PDF looks good
RUN apt-get update && apt-get install -y --no-install-recommends \
    wkhtmltopdf \
    fonts-dejavu-core \
    fonts-liberation \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501
CMD ["streamlit", "run", "00_Home.py"]
