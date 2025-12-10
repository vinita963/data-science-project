from src.datascience.config.configuration import ConfigManager
from src.datascience.componenets.model_evaluation import ModelEvaluation
from src.datascience import logger

STAGE_NAME= "Model Evaluation Stage"

class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass

    def initiate_model_evaluation(self):
        config = ConfigManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation = ModelEvaluation(config=model_evaluation_config)
        model_evaluation.log_into_mlflow()