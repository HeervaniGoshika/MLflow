import mlflow
from mlflow import MlflowClient

# Tracking DB
mlflow.set_tracking_uri("sqlite:///mlflow.db")

client = MlflowClient()

experiment_name = "wine-classification"
experiment = client.get_experiment_by_name(experiment_name)

# Get latest run from experiment
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["attributes.start_time DESC"],
    max_results=1
)

if len(runs) == 0:
    raise Exception("No runs found. Run train.py first.")

run_id = runs[0].info.run_id
model_uri = f"runs:/{run_id}/model"
model_name = "WineClassifier"

# Create registered model (first time only)
try:
    client.create_registered_model(model_name)
except:
    pass  # model already exists

# Create model version
model_version = client.create_model_version(
    name=model_name,
    source=model_uri,
    run_id=run_id
)

print(f"Model Version {model_version.version} created successfully.")
