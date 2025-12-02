from mlflow import MlflowClient
import mlflow

# Connect to MLflow Tracking
mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

model_name = "WineClassifier"

# Fetch all versions of the model
versions = client.search_model_versions(f"name='{model_name}'")

print("Available model versions:")
for v in versions:
    print(f"Version={v.version}, Stage={v.current_stage}")

# If no versions exist → stop
if not versions:
    raise Exception(f"No versions found for model {model_name}")

# Sort by version number
versions_sorted = sorted(versions, key=lambda x: int(x.version))

latest_version = versions_sorted[-1].version  # highest version number

# ---- Promote latest version to Staging ----
client.transition_model_version_stage(
    name=model_name,
    version=latest_version,
    stage="Staging"
)
print(f"Version {latest_version} → Staging")

# ---- Promote first version to Production (example logic) ----
first_version = versions_sorted[0].version

client.transition_model_version_stage(
    name=model_name,
    version=first_version,
    stage="Production"
)
print(f"Version {first_version} → Production")

# ---- Archive any remaining versions ----
for v in versions_sorted[1:-1]:   # skip first & last
    client.transition_model_version_stage(
        name=model_name,
        version=v.version,
        stage="Archived"
    )
    print(f"Version {v.version} → Archived")

print("\nModel stage updates complete!")
