FROM python:3.11-slim

WORKDIR /app

# ---- System deps + Docker CLI + Compose plugin (v2) ----
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      gnupg \
      build-essential; \
    install -m 0755 -d /etc/apt/keyrings; \
    curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg; \
    chmod a+r /etc/apt/keyrings/docker.gpg; \
    CODENAME="$(. /etc/os-release && echo "${VERSION_CODENAME}")"; \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian ${CODENAME} stable" > /etc/apt/sources.list.d/docker.list; \
    apt-get update; \
    apt-get install -y --no-install-recommends docker-ce-cli docker-compose-plugin; \
    rm -rf /var/lib/apt/lists/*

# ---- Python deps ----
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Project ----
COPY . .
ENV PYTHONPATH=/app

CMD ["uvicorn", "sentinelx.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
