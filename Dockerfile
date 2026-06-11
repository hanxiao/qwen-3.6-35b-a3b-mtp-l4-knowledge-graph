FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    httpx \
    numpy \
    python-multipart \
    pypdf \
    sentence-transformers \
    peft \
    einops \
    torch --extra-index-url https://download.pytorch.org/whl/cpu

COPY app.py .

EXPOSE 3000

CMD ["python", "app.py"]
