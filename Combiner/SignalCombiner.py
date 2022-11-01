from src.common.logger import get_logger

logger = get_logger()

SERVICE_NAME = 'Combiner'


class SignalCombiner:
    def __init__(self):
        logger.info(f"Initializing {SERVICE_NAME} service")

    def aggregate(self, features, names=[], meta=[]):
        logger.info(f"model features: {features}")
        logger.info(f"model names: {names}")
        logger.info(f"model meta: {meta}")
        return [{'combiner': features}]
        # return [x for x in features]
