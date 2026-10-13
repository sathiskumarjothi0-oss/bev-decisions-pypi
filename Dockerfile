# Bev HTTP server on CPU (Bev-Large models). For Bev-Pro models on an NVIDIA GPU use docker/Dockerfile.cuda.
#   docker build -t bev .
#   docker run -p 8000:8000 -v hf-cache:/root/.cache/huggingface bev
FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir ".[serve]"
ENV BEV_MODEL=sathiskumarjothi0/Bev-Large-C
EXPOSE 8000
CMD ["sh", "-c", "bev serve --host 0.0.0.0 --port 8000 --model $BEV_MODEL"]
