FROM python:3.8-slim

COPY TextSummary/requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install -r requirements.txt --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pypi.python.org
RUN pip uninstall -y Flask && pip install Flask --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pypi.python.org

ENV MODEL_NAME Summarization
ENV SERVICE_TYPE MODEL
ENV PERSISTENCE 0

COPY TextSummary /app

CMD exec seldon-core-microservice $MODEL_NAME  --service-type $SERVICE_TYPE --persistence $PERSISTENCE
