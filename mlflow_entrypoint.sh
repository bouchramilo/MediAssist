#!/bin/sh
set -e

# Run database migrations
echo "Running MLflow database migrations..."
mlflow db upgrade $MLFLOW_BACKEND_STORE_URI

# Start the MLflow server
echo "Starting MLflow server..."
exec "$@"
