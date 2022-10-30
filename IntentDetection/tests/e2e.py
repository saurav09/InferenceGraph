import json

import numpy as np
import requests
from seldon_core.seldon_client import SeldonClient

input_text = [
        {"item_id": 1,
         "text": "India vs South Africa Highlights, T20 World Cup 2022: South Africa beat India by 5 wickets, jump to top of the standings"},
    ]


def direct_request_test():
    """
    Tests seldon core microservice deployment via a direct request
    :return: None
    """
    url = "http://localhost:9000/predict"
    payload = {"data": {"ndarray": input_text}}

    session = requests.Session()
    session.trust_env = False

    response = requests.post(url, data={"json": json.dumps(payload)})
    print(f"Response from the direct request to the model is: {response.json()}")


def seldon_client_test():
    """
    Tests seldon core microservice deployment via request sent using SeldonClient
    :return: None
    """
    url = "localhost:9000"
    data = np.array(input_text)

    sc = SeldonClient(microservice_endpoint=url)
    response = sc.microservice(data=data, method="predict", payload_type="ndarray")
    print(f"Response from the SeldonClient's request to the model is: {response}")


direct_request_test()
seldon_client_test()
