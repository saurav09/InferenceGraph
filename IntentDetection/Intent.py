import os
from typing import List, Dict

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class Intent(object):

    def __init__(self):
        model_path = f'{os.path.dirname(os.path.abspath(__file__))}/model/t5-base-finetuned-e2m-intent'

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True)

    def predict(self, request_body: List[Dict], features_names=None) -> List[Dict]:
        intent_texts = []
        for request in request_body:
            intent_texts.append(request.get('text'))

        features = self.tokenizer(intent_texts, return_tensors='pt', padding=True)

        outputs = self.model.generate(input_ids=features['input_ids'],
                                      attention_mask=features['attention_mask'],
                                      max_length=5)

        response = []

        for request, intent_text in zip(request_body, outputs):
            response.append({**request, **{'intent': self.tokenizer.decode(intent_text)}})

        return response


if __name__ == '__main__':
    request_body = [
        {"item_id": 1,
         "text": "India vs South Africa Highlights, T20 World Cup 2022: South Africa beat India by 5 wickets, jump to top of the standings"},
    ]

    output_all_fields = Intent().predict(request_body)
    print(output_all_fields)
