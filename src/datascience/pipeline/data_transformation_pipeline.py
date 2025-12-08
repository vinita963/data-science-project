from src.datascience.config.configuration import ConfigManager
from src.datascience.componenets.data_transformation import DataTransformation
from src.datascience import logger
from pathlib import Path

STAGE_NAME= "Data Transformation Stage"

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def initiate_data_transformation(self):
        try:
            with open(Path("artifacts/data_validation/status.txt"), 'r') as f:
                status=f.read().split(" ")[-1]

            if status == "True":
                config = ConfigManager()
                data_transformation_config=config.get_data_transformation_config()
                data_transformation_config=DataTransformation(config=data_transformation_config)
                data_transformation= DataTransformation(config=data_transformation_config)
                data_transformation.train_test_splitting()
            else:
                raise Exception("data schema is not valid")
            
        except Exception as e:
            print(e)

