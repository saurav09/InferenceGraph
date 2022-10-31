import os
from typing import List, Dict

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

from src.common.logger import get_logger
from src.common.timer import timeit

logger = get_logger()


class Summarization:
    """
    Translation Model template. You can load your model parameters in __init__ from a location accessible at runtime
    """

    def __init__(self):
        logger.info('initialising Translation service')
        self.loaded = False
        logger.debug(f'model load status is {self.loaded}')

    @timeit
    def _load(self):
        logger.info("loading the model")

        self.device = self._get_device()
        logger.info(f"detected device is: {self.device}")
        model_path = f'{os.path.dirname(os.path.abspath(__file__))}/model/t5-small'
        self.model = T5ForConditionalGeneration.from_pretrained(model_path, local_files_only=True)
        self.tokenizer = T5Tokenizer.from_pretrained(model_path, local_files_only=True)

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
    def predict(self, request_body: List[Dict], features_names=None) -> List[Dict]:
        if not self.loaded:
            self._load()

        summarized_texts = []

        for request in request_body:
            inputs = self.tokenizer.encode("summarize: " + request.get('text'), return_tensors="pt", max_length=512,
                                           truncation=True)
            outputs = self.model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=1,
                                          early_stopping=True)
            summarized_texts.append({
                "item_id": request.get('item_id'),
                "summary": self.tokenizer.decode(outputs[0]),
            })
        return summarized_texts


if __name__ == '__main__':
    original_text = """
    Paul Walker is hardly the first actor to die during a production. 
    But Walker's death in November 2013 at the age of 40 after a car crash was especially eerie given his rise to fame in the "Fast and Furious" film franchise. 
    The release of "Furious 7" on Friday offers the opportunity for fans to remember -- and possibly grieve again -- the man that so many have praised as one of the nicest guys in Hollywood. 
    "He was a person of humility, integrity, and compassion," military veteran Kyle Upham said in an email to CNN. 
    Walker secretly paid for the engagement ring Upham shopped for with his bride. 
    "We didn't know him personally but this was apparent in the short time we spent with him. 
    I know that we will never forget him and he will always be someone very special to us," said Upham. 
    The actor was on break from filming "Furious 7" at the time of the fiery accident, which also claimed the life of the car's driver, Roger Rodas. 
    Producers said early on that they would not kill off Walker's character, Brian O'Connor, a former cop turned road racer. Instead, the script was rewritten and special effects were used to finish scenes, with Walker's brothers, Cody and Caleb, serving as body doubles. 
    There are scenes that will resonate with the audience -- including the ending, in which the filmmakers figured out a touching way to pay tribute to Walker while "retiring" his character. At the premiere Wednesday night in Hollywood, Walker's co-star and close friend Vin Diesel gave a tearful speech before the screening, saying "This movie is more than a movie." "You'll feel it when you see it," Diesel said. "There's something emotional that happens to you, where you walk out of this movie and you appreciate everyone you love because you just never know when the last day is you're gonna see them." There have been multiple tributes to Walker leading up to the release. Diesel revealed in an interview with the "Today" show that he had named his newborn daughter after Walker. 
    Social media has also been paying homage to the late actor. A week after Walker's death, about 5,000 people attended an outdoor memorial to him in Los Angeles. Most had never met him. Marcus Coleman told CNN he spent almost $1,000 to truck in a banner from Bakersfield for people to sign at the memorial. "It's like losing a friend or a really close family member ... even though he is an actor and we never really met face to face," Coleman said. "Sitting there, bringing his movies into your house or watching on TV, it's like getting to know somebody. It really, really hurts." Walker's younger brother Cody told People magazine that he was initially nervous about how "Furious 7" would turn out, but he is happy with the film. "It's bittersweet, but I think Paul would be proud," he said. CNN's Paul Vercammen contributed to this report.
    """

    request_body = [
        {"item_id": 1, "text": original_text},
        {"item_id": 2, "text": original_text}
    ]

    output_all_fields = Summarization().predict(request_body)
    print(output_all_fields)
