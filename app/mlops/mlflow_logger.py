import mlflow
from mlflow.tracking import MlflowClient
from typing import Dict, Any, Optional

class MLflowLogger:

    def __init__(self, experiment_name: str = "RAG_Experiment", run_id: Optional[str] = None):
        self.client = MlflowClient()
        self.run_id = run_id
        if not run_id:
             mlflow.set_experiment(experiment_name)
             self.experiment = mlflow.get_experiment_by_name(experiment_name)

    def start_run(self, run_name: str):
        run = mlflow.start_run(run_name=run_name)
        self.run_id = run.info.run_id
        return run

    def end_run(self):
        mlflow.end_run()

    def log_rag_config(self, config: Dict[str, Any]):
        """
        Log configuration complète du pipeline RAG
        """
        if self.run_id:
            for k, v in config.items():
                self.client.log_param(self.run_id, k, str(v))
        else:
            mlflow.log_params(config)

    def log_metrics(self, metrics: Dict[str, float]):
        if self.run_id:
            for k, v in metrics.items():
                self.client.log_metric(self.run_id, k, v)
        else:
            mlflow.log_metrics(metrics)

    def log_artifact(self, file_path: str):
        if self.run_id:
            self.client.log_artifact(self.run_id, file_path)
        else:
            mlflow.log_artifact(file_path)

    def log_text(self, text: str, artifact_file: str):
        if self.run_id:
            self.client.log_text(self.run_id, text, artifact_file)
        else:
            mlflow.log_text(text, artifact_file)


logger = MLflowLogger(experiment_name="RAG_MediAssist")