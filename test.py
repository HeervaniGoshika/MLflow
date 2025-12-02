from mlflow import MlflowClient

client = MlflowClient()

model_name = "WineClassifier"
versions = client.get_latest_versions(model_name)

print("Available Versions:")
for v in versions:
    print(f"Version: {v.version}, Stage: {v.current_stage}")
