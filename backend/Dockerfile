# Backend image for Hugging Face Spaces (Docker SDK). Build context = the contents of backend/.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# CPU-only PyTorch is far smaller than the default CUDA build.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Download CLIP and pre-build the image index at build time so the first request is fast.
RUN python -c "from transformers import CLIPModel, CLIPProcessor; n='openai/clip-vit-base-patch32'; CLIPProcessor.from_pretrained(n); CLIPModel.from_pretrained(n)" \
 && python -c "import main; print('indexed', main.rebuild_index())" \
 && chmod -R 777 /app

EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
