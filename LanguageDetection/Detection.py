import os
from typing import Dict, List

import fasttext

from src.common.logger import get_logger
from src.common.timer import timeit

logger = get_logger()


class Detection:
    def __init__(self):
        logger.info(f'Initialising Language Detection service')
        self.model = fasttext.load_model(
            f'{os.path.dirname(os.path.abspath(__file__))}/model/fasttext/lid.176.bin')
        logger.info(f'Language Detection model is loaded')

    @timeit
    def predict(self, request_body: List[Dict], features_names=None) -> List[Dict]:
        results = []
        for request in request_body:
            detected_language, detected_language_probability = None, None
            try:
                prediction_response = self.model.predict(request.get('text'), k=3)
                detected_language = prediction_response[0][0].split('__')[-1]
                detected_language_probability = round(prediction_response[1][0], 2)
            except:
                logger.error(f"failed to detect language for the item {request.get('item_id')}")

            response = {
                'item_id': request.get('item_id'),
                'text': request.get('text'),
                'detected_language': detected_language,
                'detected_language_probability': detected_language_probability
            }
            results.append(response)
        return results


if __name__ == '__main__':
    requests = [{
        'item_id': "abc123",
        'text': "When will I get the refund?"
    }, {
        'item_id': "abc234",
        'text': ""
    }]
    r = Detection().predict(requests)
    print(r)
