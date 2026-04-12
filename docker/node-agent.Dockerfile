FROM vllm/vllm-openai:latest

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

RUN pip install --no-cache-dir fastapi "uvicorn[standard]" httpx pydantic ray pynvml

COPY src /app/src

EXPOSE 9000

ENTRYPOINT ["python", "-m", "axon.node_agent"]
