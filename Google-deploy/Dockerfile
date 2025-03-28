FROM python:3.11-slim

# Install system packages required by LightGBM, XGBoost, and clean up
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=True

# Copy local code to the image
ENV APP_HOME=/app
WORKDIR $APP_HOME
COPY . .

RUN pip install -r requirements.txt

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app