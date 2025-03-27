FROM python:3.11-slim

ENV PYTHONUNBUFFERED=True
WORKDIR /app
COPY . .

RUN pip install -r requirements.txt

CMD ["sh", "-c", "python scrape_avgs_update.py && python scrape_rec_update.py && python inference.py"]
