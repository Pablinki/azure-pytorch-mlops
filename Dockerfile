# Serving image: CPU torch, multi-stage, non-root. The model is baked in (immutable artifact, fast cold start).
FROM python:3.11-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install ".[serve]"

FROM python:3.11-slim AS runtime
RUN useradd --create-home --uid 10001 appuser
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY configs ./configs
# Produced by `make docker-build` (copied from outputs/model) or by CI (downloaded registered model).
COPY model ./model
ENV LOG_FORMAT=json TEXTCLF_HOST=0.0.0.0 TEXTCLF_PORT=8000 TEXTCLF_MODEL_URI=file:///app/model
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
  CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health')" || exit 1
CMD ["textclf", "serve"]
