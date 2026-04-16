ARG VLLM_BASE_IMAGE=vllm/vllm-openai:latest
FROM ${VLLM_BASE_IMAGE}

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/node/src

COPY requirements-base.txt /app/requirements-base.txt
COPY node/requirements.txt /app/node/requirements.txt
RUN pip install --no-cache-dir -r /app/node/requirements.txt

COPY node/src /app/node/src

EXPOSE 9000

ENTRYPOINT ["python", "/app/node/src/agent.py"]
