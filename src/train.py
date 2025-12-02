import mlflow
import mlflow.sklearn
import joblib
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


mlflow.set_tracking_uri("sqlite:///mlflow.db")

# 1. Set experiment
mlflow.set_experiment("wine-classification")  #track experiment

# 2. Load data
X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 3. Start run
with mlflow.start_run():

    # ----- Track parameters -----
    n_estimators = 100
    depth = 5

    mlflow.log_param("n_estimators", n_estimators)    #track parameteres
    mlflow.log_param("max_depth", depth)

    #mlflow.log_param("n_estimators", 100)
    #mlflow.log_metric("accuracy", acc)

    # Train model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=depth
    )
    model.fit(X_train, y_train)

    # Predictions
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # ----- Track metrics -----
    mlflow.log_metric("accuracy", acc)     #track metrics

    # ----- Log model -----
    mlflow.sklearn.log_model(model, name="model")

    print("Accuracy:", acc)

model = SVC()
model.fit(X_train, y_train)

joblib.dump(model, "model.pkl")
