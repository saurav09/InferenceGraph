import os
import traceback
from typing import List, Dict

import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

from src.common.logger import get_logger
from src.common.timer import timeit

logger = get_logger()

SERVICE_NAME = 'Text Translation'


class Translation:
    """
    Translation Model template. You can load your model parameters in __init__ from a location accessible at runtime
    """

    def __init__(self):
        logger.info(f'initialising {SERVICE_NAME} service')
        self.loaded = False
        logger.debug(f'model load status is {self.loaded}')

    @timeit
    def _load(self):
        logger.info("loading the model")

        self.device = self._get_device()
        logger.info(f"detected device is: {self.device}")
        model_path = f'{os.path.dirname(os.path.abspath(__file__))}/model/m2m100_418M'
        self.model = M2M100ForConditionalGeneration.from_pretrained(model_path, local_files_only=True).to(self.device)
        self.tokenizer = M2M100Tokenizer.from_pretrained(model_path, local_files_only=True)

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

    def _translate(self, text: List[str], source_lang: str, target_lang: str) -> List[str]:
        try:
            self.tokenizer.src_lang = source_lang

            encoded_src = self.tokenizer(text,
                                         return_tensors="pt",
                                         padding=True,
                                         truncation=True,
                                         max_length=1024)

            generated_tokens = self.model.generate(**encoded_src,
                                                   forced_bos_token_id=self.tokenizer.get_lang_id(target_lang),
                                                   num_beams=1,
                                                   no_repeat_ngram_size=5,
                                                   repetition_penalty=1.2)

            translated_texts = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        except RuntimeError as error:
            if str(error).startswith('CUDA out of memory.'):
                raise Exception("CUDA out of memory")
            else:
                raise Exception(error)

        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return translated_texts

    @timeit
    def predict(self, request_body: List[Dict], features_names=None) -> List[Dict]:
        """
        Returns a prediction.

        Parameters
        ----------
        request_body : array-like
        feature_names : array of feature names (optional)
        """

        # Model is loaded once in the predict call and not in the init due to -
        # https://docs.seldon.io/projects/seldon-core/en/latest/python/python_component.html?highlight=load#gunicorn-and-load
        # https://github.com/SeldonIO/seldon-core/issues/2220

        logger.info(f"Request in {SERVICE_NAME} service is {request_body}")

        if not self.loaded:
            self._load()

        texts_to_translate = []
        translation_response = []
        try:
            request_body = request_body[0]
            source_lang = request_body.get('source_language')
            target_lang = request_body.get('target_language')

            for request in request_body.get('translation_items'):
                texts_to_translate.append(request.get('text'))

            translated_items = self._translate(texts_to_translate, source_lang, target_lang)

            for item, translation in zip(request_body.get('translation_items'), translated_items):
                translation_response.append({**{'item_id': item.get('item_id')}, **{'translated_text': translation}})
        except Exception as e:
            logger.error(f'error in processing the request, {e}, {traceback.format_exc()}')
            return []

        logger.info(f"Response from {SERVICE_NAME} service is {translation_response}")
        return translation_response


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

    output_all_fields = Translation().predict(request_all_fields)
    print(f'Prediction (all input fields present):\n {output_all_fields}')
