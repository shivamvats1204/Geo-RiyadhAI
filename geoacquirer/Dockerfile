FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
ENV GEOACQUIRER_CACHE=/data/cache
VOLUME ["/data"]
EXPOSE 8000
CMD ["geoacquirer", "serve", "--host", "0.0.0.0", "--port", "8000", "--out", "/data/runs"]
