
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY monitor_app.py .

COPY deployment_model/ ./deployment_model/

ENV MODEL_URI=/app/deployment_model

EXPOSE 5003

CMD ["python", "monitor_app.py"]
