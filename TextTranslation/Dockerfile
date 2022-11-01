FROM python:3.8-slim

COPY TextTranslation/requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install -r requirements.txt --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pypi.python.org
RUN pip uninstall -y Flask && pip install Flask --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pypi.python.org

ENV MODEL_NAME Translation
ENV SERVICE_TYPE MODEL
ENV PERSISTENCE 0

COPY TextTranslation /app

CMD exec seldon-core-microservice $MODEL_NAME  --service-type $SERVICE_TYPE --persistence $PERSISTENCE
