
# Minimal MLflow logger wrapper
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except:
    MLFLOW_AVAILABLE = False

def log_run(run_name, metrics, artifact_paths):
    if not MLFLOW_AVAILABLE:
        print("MLflow not installed; skipping logging.")
        return
    mlflow.set_experiment("form-eval")
    with mlflow.start_run(run_name=run_name):
        for k,v in metrics.items():
            mlflow.log_metric(k, v)
        for ap in artifact_paths:
            mlflow.log_artifact(ap)
