from src.datascience.config.configuration import ConfigManager
from src.datascience.componenets.model_trainer import ModelTrainer
from src.datascience import logger

STAGE_NAME= "Model Trainer Stage"

class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def initiate_model_training(self):
        config = ConfigManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer_config = ModelTrainer(config=model_trainer_config)
        model_trainer_config.train()