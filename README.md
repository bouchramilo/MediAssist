# 🏥 MediAssist Pro – Assistant Cognitif de Maintenance Biomédicale

## 📌 Description du projet

**MediAssist Pro** est un assistant intelligent basé sur une architecture **RAG (Retrieval-Augmented Generation)** conçu pour aider les équipes de laboratoire à résoudre rapidement les incidents techniques liés aux équipements biomédicaux.

Le système indexe les manuels techniques (PDF) et les bases de connaissances internes afin de fournir des réponses :

- ✅ Précises  
- ✅ Contextualisées  
- ✅ Sourcées  
- ✅ Actionnables  

L’objectif est de réduire les délais d’intervention, limiter l’ouverture de tickets support et assurer la continuité des analyses biomédicales.

Le projet intègre une architecture complète **LLMOps & MLOps**, incluant :
- Tracking & évaluation RAG avec **MLflow**
- Évaluation qualité avec **DeepEval**
- API sécurisée via **FastAPI + JWT**
- CI/CD automatisée
- Déploiement conteneurisé (Docker)
- Monitoring avec **Prometheus & Grafana**

---

# 🧠 Architecture RAG

## 1️⃣ Ingestion & Prétraitement

- Chargement des documents PDF techniques
- Nettoyage et segmentation intelligente (chunking avec overlap)
- Ajout de métadonnées (source, section, page)

## 2️⃣ Vectorisation & Indexation

- Génération d’embeddings (LLM / Hugging Face)
- Stockage dans une base vectorielle persistante (**ChromaDB**)
- Persistance des embeddings

## 3️⃣ Retrieval

- Recherche sémantique par similarité (cosine)
- Configuration du nombre de chunks retournés (k)
- Reranking et amélioration des requêtes

## 4️⃣ Génération (LLM)

- Prompt engineering centralisé
- Génération contextualisée à partir des chunks récupérés
- Réduction des hallucinations
- Réponses exclusivement fondées sur les documents indexés

---

# 🔐 Fonctionnalités principales

## 🌐 API REST (FastAPI)

- Endpoint `/chat` : interaction avec l’assistant
- Endpoint `/documents` : gestion des documents
- Endpoint `/admin` : administration
- Validation des données avec **Pydantic**
- Documentation automatique via Swagger

## 🔐 Sécurité

- Authentification JWT
- Hashage sécurisé des mots de passe
- Gestion centralisée des exceptions
- Configuration via `.env`

## 🗄️ Base de données

- **PostgreSQL**
- ORM : **SQLAlchemy**
- Tables principales :
  - `users`
  - `query`

## 📊 LLMOps & Tracking

Avec **MLflow** :

- Logging configuration RAG :
  - Taille des chunks & overlap
  - Modèle d’embeddings
  - Paramètres retrieval (k, similarité)
  - Paramètres LLM (température, max_tokens, top_p, etc.)
- Logging des réponses & contextes
- Logging des métriques RAG :
  - Answer Relevance
  - Faithfulness
  - Precision@k
  - Recall@k
- Tracking du pipeline LangChain

Évaluation automatique via **DeepEval**

---

# 📈 Monitoring & Observabilité

## Prometheus

- Collecte des métriques applicatives :
  - Latence
  - Nombre de requêtes
  - Taux d’erreurs
  - Qualité des réponses
- Métriques infrastructure :
  - CPU
  - RAM
  - Statut du Pod

## Grafana

- Dashboard dédié MediAssist Pro
- Visualisation temps réel
- Alertes configurables (latence, erreurs, qualité)

---

# 🔄 CI/CD & Déploiement

## GitHub Actions

- Exécution automatique des tests
- Validation du pipeline RAG
- Build image Docker
- Publication sur Docker Hub

## Kubernetes (Minikube)

- Déploiement en Pod unique
- Gestion via `service.yml` et `deployment.yml`
- Supervision du Pod

---

# 🗂️ Structure du projet

```bash
.
├── app/
│   ├── api/              # Endpoints FastAPI
│   ├── services/         # Logique RAG (chunking, retriever, LLM, embeddings)
│   ├── mlops/            # Tracking MLflow & évaluation DeepEval
│   ├── models/           # Modèles SQLAlchemy
│   ├── schemas/          # Schémas Pydantic
│   ├── repositories/     # Couche accès base de données
│   ├── security/         # JWT & hash mots de passe
│   ├── config/           # Configuration & gestion exceptions
│   └── main.py           # Point d’entrée FastAPI
│
├── monitoring/           # Prometheus & Grafana config
├── tests/                # Tests unitaires (≥ 80% coverage)
├── vector_store/         # Stockage ChromaDB
├── Dockerfile            # Image principale
├── Dockerfile.mlflow     # Image dédiée MLflow
├── docker-compose.yml    # Orchestration locale
├── deployement.yml       # Déploiement Kubernetes
└── service.yml           # Service Kubernetes
```

---

# 🛠️ Technologies utilisées

* Python
* FastAPI
* LangChain
* PostgreSQL
* SQLAlchemy
* ChromaDB
* MLflow
* DeepEval
* JWT
* Docker & Docker Compose
* Kubernetes (Minikube)
* Prometheus
* Grafana
* Pytest

---

# ⚙️ Installation & Exécution (Local)

## 1️⃣ Cloner le projet

```bash
git clone https://github.com/bouchramilo/MediAssist.git
cd MediAssist
```

## 2️⃣ Configurer les variables d’environnement
```bash
cp .env.example .env
```
Modifier les variables si nécessaire.

## 3️⃣ Lancer l’infrastructure
```bash
docker-compose up --build
```


---

Merci 😊