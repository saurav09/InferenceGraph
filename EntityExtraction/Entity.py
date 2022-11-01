import os
import traceback
from typing import List, Dict

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

from src.common.logger import get_logger
from src.common.timer import timeit

logger = get_logger()

SERVICE_NAME = 'Entity Extraction'


class Entity:
    """
    Translation Model template. You can load your model parameters in __init__ from a location accessible at runtime
    """

    def __init__(self):
        logger.info(f'Initialising {SERVICE_NAME} service')
        self.loaded = False
        logger.debug(f'model load status is {self.loaded}')

    @timeit
    def _load(self):
        logger.info("loading the model")

        self.device = self._get_device()
        logger.info(f"detected device is: {self.device}")
        model_path = f'{os.path.dirname(os.path.abspath(__file__))}/model/bert-base-NER'
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path, local_files_only=True)
        self.loaded = True
        logger.debug('model is loaded')

    @staticmethod
    def _get_device():
        if not torch.cuda.is_available():
            return "cpu"

        num_cuda = torch.cuda.device_count()
        _id = os.getpid()

        _device_ids = list(range(0, num_cuda))
        current_device = _device_ids[((_id + 1) % len(_device_ids))]

        return f"cuda:{current_device}"

    @timeit
    def predict(self, request_body: List[Dict], features_names=None) -> Dict:
        logger.info(f"Request in {SERVICE_NAME} service is {request_body}")
        try:
            if not self.loaded:
                self._load()

            entities_from_texts = []

            for request in request_body:
                nlp = pipeline("ner", model=self.model, tokenizer=self.tokenizer)
                entities_from_texts.append({
                    "item_id": request.get('item_id'),
                    "entities": str(nlp(request.get('translated_text'))),
                })
            logger.info(f"response from {SERVICE_NAME} is {entities_from_texts}")
            return {"entity": entities_from_texts}
        except Exception as e:
            traceback.print_exc()
            return []


if __name__ == '__main__':
    request_body = [
        {"item_id": 1, "translated_text": "I live in India and I love mu country"},
        {"item_id": 2, "translated_text": "My name is Wolfgang and I live in Berlin"}
    ]

    output_all_fields = Entity().predict(request_body)
    print(output_all_fields)
