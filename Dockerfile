FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python scripts/generate_sample_data.py \
    && python scripts/process_dataset.py \
    && python scripts/train_salary_model.py \
    && python -m src.utils.db

EXPOSE 8501 8000

CMD ["sh", "-c", "uvicorn app.api:app --host 0.0.0.0 --port 8000 & streamlit run app/dashboard.py --server.port=8501 --server.address=0.0.0.0"]
