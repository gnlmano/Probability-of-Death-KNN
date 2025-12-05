FROM python:3.11

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

ENV PYTHONPATH="${PYTHONPATH}:/app/src"

EXPOSE 8000

CMD ["uvicorn", "src.probability_of_death.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
