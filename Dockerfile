FROM python:3.12-slim

# TORCH_INDEX_URL を指定すると CPU 専用などの軽い torch を入れられる
#   例: --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
# 未指定なら PyPI 標準 (x86_64 は CUDA 同梱 / arm64 は CPU)
ARG TORCH_INDEX_URL=""

ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 \
    HF_HOME=/models OUTPUT_DIR=/outputs HF_HUB_DISABLE_XET=1 U2NET_HOME=/models/rembg

WORKDIR /app
RUN pip install --upgrade pip && \
    if [ -n "$TORCH_INDEX_URL" ]; then pip install torch --index-url "$TORCH_INDEX_URL"; else pip install torch; fi
RUN pip install "diffusers>=0.32" "transformers>=4.45" "accelerate>=1.0" "safetensors>=0.4" \
                "sentencepiece>=0.2" "protobuf>=4.25" "pillow>=10" "fastapi>=0.115" \
                "uvicorn[standard]>=0.30" "huggingface_hub>=0.25" \
                "rembg>=2.0" "onnxruntime>=1.17"

COPY server ./server
COPY doctor.py ./
RUN mkdir -p /models /outputs
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s CMD python -c "import urllib.request,json,sys;sys.exit(0 if json.load(urllib.request.urlopen('http://localhost:8000/health'))['ok'] else 1)"
# 起動前に環境診断の要約を出す (失敗しても起動は続ける)
CMD ["sh", "-c", "python doctor.py --brief || true; exec uvicorn server.app:app --host 0.0.0.0 --port 8000"]
