FROM dockerhub.paypalcorp.com/cspds/python3.7-slim

COPY LanguageDetection/requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install -r requirements.txt --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pypi.python.org
RUN pip uninstall -y Flask && pip install Flask --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pypi.python.org

EXPOSE 5000

ENV MODEL_NAME Detection
ENV SERVICE_TYPE MODEL
ENV PERSISTENCE 0

COPY LanguageDetection /app

CMD exec seldon-core-microservice $MODEL_NAME  --service-type $SERVICE_TYPE --persistence $PERSISTENCE
