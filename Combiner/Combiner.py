from src.common.logger import get_logger

logger = get_logger()

SERVICE_NAME = 'Combiner'


class Combiner:
    def __init__(self):
        logger.info(f"Initializing {SERVICE_NAME} service")

    def aggregate(self, features, names=[], meta=[]):
        return [x for x in features]
