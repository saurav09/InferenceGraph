from src.common.logger import get_logger

logger = get_logger()


class Router:
    def __init__(self):
        logger.info("Initializing router service")
        print("Initializing router service")

    def route(self, features, feature_names=None) -> int:
        logger.info(f"Request in Router is {features}")
        print(f"Request in Router is {features}")
        return -1


if __name__ == '__main__':
    request_all_fields = [{
        "source_language": "en",
        "target_language": "hi",
        "translation_items": [
            {
                "item_id": "1",
                "text": "English is a language"
            },
            {
                "item_id": "2",
                "text": "We will extend the conventions laid out to accommodate this for sure."
            },
        ]
    }]
    print(Router().route(request_all_fields))
