import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import numpy as np
import joblib
import pandas as pd

os.environ["MLFLOW_TRACKING_URI"]="https://dagshub.com/vinitaraorane17849/data-science-project.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"]="vinitaraorane17849"
os.environ["MLFLOW_TRACKING_PASSWORD"]="f5e59444dcc29a43e2657e8d0581f5da19752de3"

from src.datascience.entity.config_entity import ModelEvaluationConfig
from src.datascience.constants import *
import tempfile

class ModelEvaluation:
    def __init__(self,config:ModelEvaluationConfig):
        self.config=config

    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2
    
    
    def log_into_mlflow(self):
        mlflow.autolog(disable=True)

        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[self.config.target_column]

        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

        with mlflow.start_run():
            preds = model.predict(test_x)
            rmse, mae, r2 = self.eval_metrics(test_y, preds)

            mlflow.log_params(self.config.all_params)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)

            # ✅ SAFE WAY: manual artifact logging
            with tempfile.TemporaryDirectory() as tmpdir:
                model_path = os.path.join(tmpdir, "model.joblib")
                joblib.dump(model, model_path)
                mlflow.log_artifact(model_path, artifact_path="model")

  

    run_id = "beb92f027aa8476fb581c787562652a6"
    model_path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path="model/model.joblib"
    )

    model = joblib.load(model_path)