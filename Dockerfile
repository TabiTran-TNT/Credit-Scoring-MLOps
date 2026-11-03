FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT=true EVIDENTLY_DISABLE_TELEMETRY=1
WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && useradd --uid 1000 --create-home app \
    && mkdir -p logs work reports model data deployments .mlflow \
    && chown -R app:app /workspace
COPY --chown=app:app app app
COPY --chown=app:app credit_scoring credit_scoring
COPY --chown=app:app monitoring monitoring
COPY --chown=app:app scripts scripts
COPY --chown=app:app optimization optimization
COPY --chown=app:app configs configs
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
