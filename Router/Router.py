from src.common.logger import get_logger

logger = get_logger()


class Router:
    def __init__(self):
        logger.info("Initializing router service")

    def route(self, features, feature_names=None) -> int:
        logger.info(f"Request in Router is {features}")
        return 0