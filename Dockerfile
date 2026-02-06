FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app

RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copier d'abord les fichiers nécessaires au build
COPY app/ ./app/
COPY data/ ./data/

# Copier les autres fichiers seulement si nécessaires
COPY .env.example .env.example
# COPY .dockerignore .dockerignore
COPY docker-compose.yml docker-compose.yml
COPY README.md README.md

# Exposer le port
EXPOSE 8000

# Commande de démarrage
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]