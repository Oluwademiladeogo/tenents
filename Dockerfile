FROM python:3.10-slim

WORKDIR /app

RUN ls -lR /app

COPY . .

RUN ls -lR /app

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/logs

EXPOSE 5000

ENV PYTHONPATH="${PYTHONPATH}:/app"

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "5000"] 