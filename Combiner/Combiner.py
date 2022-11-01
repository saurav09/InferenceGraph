from src.common.logger import get_logger

logger = get_logger()

SERVICE_NAME = 'Combiner'


class Combiner:
    def __init__(self):
        logger.info(f"Initializing {SERVICE_NAME} service")

    def aggregate(self, features, feature_names=None):
        logger.info(f"Request in {SERVICE_NAME} service is {features}")

        return [x for x in features]
