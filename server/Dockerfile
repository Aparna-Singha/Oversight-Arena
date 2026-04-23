FROM python:3.11-slim

WORKDIR /app

COPY server/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared models first (needed by server modules)
COPY models.py /app/models.py

# Copy server source
COPY server/ /app/

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
