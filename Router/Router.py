from src.common.logger import get_logger

logger = get_logger()

SERVICE_NAME = 'Router'


class Router:
    def __init__(self):
        logger.info(f"Initializing {SERVICE_NAME} service")

    def route(self, features, feature_names=None) -> int:
        logger.info(f"Request in {SERVICE_NAME} service is {features}")
        return 1
