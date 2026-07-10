FROM python:3.11-slim

WORKDIR /app

# Install the package (pulls in pandas/scikit-learn/xgboost/fastapi/etc. from pyproject.toml).
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .

# Train and bake in the model artifact from scratch (models/ is gitignored,
# never committed -- see scripts/train_and_save.py). Needs network access at
# build time to download the SUPPORT dataset, same as `pytest -q` does locally.
COPY scripts/train_and_save.py ./scripts/train_and_save.py
RUN python scripts/train_and_save.py

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "support_survival.api:app", "--host", "0.0.0.0", "--port", "8000"]
