Inference Graph design

![Alt text](resources/graph.png?raw=True "Inference Graph")



API Specifications: 
<br>

Request Format

POST http://localhost:8000/api/v1.0/predictions
```
{
    "data": {
        "ndarray": [
            {
                "item_id": "abc1",
                "text": "मार्स ऑर्बिटर मिशन (MOM), जिसे अनौपचारिक रूप से मंगलयान के रूप में जाना जाता है, को भारतीय अंतरिक्ष अनुसंधान संगठन (ISRO) द्वारा 5 नवंबर 2013 को पृथ्वी की कक्षा में लॉन्च किया गया था और 24 सितंबर 2014 को मंगल की कक्षा में प्रवेश कर गया। इस प्रकार भारत मंगल में प्रवेश करने वाला पहला देश बन गया। अपने पहले प्रयास में कक्षा। इसे 74 मिलियन डॉलर की रिकॉर्ड कम लागत पर पूरा किया गया था।"
            }
        ]
    }
}
```

Response Format

```{
    "data": {
        "names": [],
        "ndarray": [
            {
                "intent": [
                    {
                        "intent": "to explore the universe",
                        "item_id": "abc1"
                    }
                ]
            },
            {
                "summary": [
                    {
                        "item_id": "abc1",
                        "summary": "the Munglian was launched on November 5, 2013 by the Indian Space Research Organization. India became the first country to enter the Mars orbit. the munglian was also the first country to enter the orbit."
                    }
                ]
            },
            {
                "entity": [
                    {
                        "entities": "[{'entity': 'B-ORG', 'score': 0.732966, 'index': 1, 'word': 'Mars', 'start': 0, 'end': 4}, {'entity': 'I-ORG', 'score': 0.83634466, 'index': 2, 'word': 'Or', 'start': 5, 'end': 7}, {'entity': 'I-ORG', 'score': 0.9081488, 'index': 3, 'word': '##bit', 'start': 7, 'end': 10}, {'entity': 'I-ORG', 'score': 0.80723315, 'index': 4, 'word': '##er', 'start': 10, 'end': 12}, {'entity': 'I-ORG', 'score': 0.69890326, 'index': 5, 'word': 'Mission', 'start': 13, 'end': 20}, {'entity': 'B-MISC', 'score': 0.58012897, 'index': 7, 'word': 'M', 'start': 22, 'end': 23}, {'entity': 'B-MISC', 'score': 0.9854071, 'index': 15, 'word': 'Mu', 'start': 52, 'end': 54}, {'entity': 'I-MISC', 'score': 0.42627418, 'index': 16, 'word': '##ng', 'start': 54, 'end': 56}, {'entity': 'I-MISC', 'score': 0.93398607, 'index': 17, 'word': '##lian', 'start': 56, 'end': 60}, {'entity': 'B-ORG', 'score': 0.9983663, 'index': 28, 'word': 'Indian', 'start': 102, 'end': 108}, {'entity': 'I-ORG', 'score': 0.9988661, 'index': 29, 'word': 'Space', 'start': 109, 'end': 114}, {'entity': 'I-ORG', 'score': 0.99890506, 'index': 30, 'word': 'Research', 'start': 115, 'end': 123}, {'entity': 'I-ORG', 'score': 0.9987237, 'index': 31, 'word': 'Organization', 'start': 124, 'end': 136}, {'entity': 'B-ORG', 'score': 0.99782604, 'index': 33, 'word': 'IS', 'start': 138, 'end': 140}, {'entity': 'I-ORG', 'score': 0.9988172, 'index': 34, 'word': '##RO', 'start': 140, 'end': 142}, {'entity': 'B-LOC', 'score': 0.98903394, 'index': 39, 'word': 'Mars', 'start': 160, 'end': 164}, {'entity': 'B-LOC', 'score': 0.99981314, 'index': 48, 'word': 'India', 'start': 199, 'end': 204}, {'entity': 'B-LOC', 'score': 0.9974769, 'index': 55, 'word': 'Mars', 'start': 239, 'end': 243}]",
                        "item_id": "abc1"
                    }
                ]
            }
        ]
    },
    "meta": {
        "requestPath": {
            "combiner": "combiner:1.0",
            "detection": "detection:1.0",
            "intent": "intent:1.0",
            "translation": "translation:1.0"
        }
    }
}
```

Steps to run the system:

* Bring up the deployment: kubectl -n [namespace] create -f deploy.yaml 
<br>
Note: The Models are not uploaded to the repository. It should be downloaded from the huggingface and put in the model directory in each of the services.

* Do a Port-forwarding: kubectl -n [namespace] port-forward svc/inference-graph-seldon-inference-graph 8000:8000

* Send a request: POST http://localhost:8000/api/v1.0/predictions

