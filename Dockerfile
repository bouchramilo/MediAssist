# Image unique pour éviter le surcoût du multi-stage lors du dev
FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Installation des dépendances système
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Copie des requirements
COPY requirements.txt .

# Installation optimisée avec cache PIP
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Copie du code source
COPY . .

# Création utilisateur non-root
RUN addgroup --system app && adduser --system --group app
USER app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
