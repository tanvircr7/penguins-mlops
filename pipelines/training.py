# pipelines/train_flow.py
from metaflow import FlowSpec, step
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

class TrainFlow(FlowSpec):
    @step
    def start(self):
        # Load data (replace with real path or download)
        df = pd.read_csv("data/penguins.csv")
        df = df.dropna()
        X = df.drop(columns=["label"])   # adjust column names to your dataset
        y = df["label"]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        self.next(self.train)

    @step
    def train(self):
        mlflow.set_experiment("penguins-baseline")
        with mlflow.start_run():
            model = LogisticRegression(max_iter=1000)
            model.fit(self.X_train, self.y_train)
            preds = model.predict(self.X_test)
            f1 = f1_score(self.y_test, preds, average="macro")
            mlflow.log_metric("f1_macro", f1)
            # Save model (local artifact)
            import joblib, os
            os.makedirs("artifacts", exist_ok=True)
            joblib.dump(model, "artifacts/model.joblib")
            mlflow.log_artifact("artifacts/model.joblib")
        self.next(self.end)

    @step
    def end(self):
        print("Done.")

if __name__ == "__main__":
    from metaflow import main
    main()
