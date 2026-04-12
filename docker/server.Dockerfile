FROM vllm/vllm-openai:latest

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

RUN pip install --no-cache-dir fastapi "uvicorn[standard]" httpx pydantic ray pynvml

COPY src /app/src

EXPOSE 8080

ENTRYPOINT ["python", "-m", "axon.server"]
