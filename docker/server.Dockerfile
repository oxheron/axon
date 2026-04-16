FROM python:3.10-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/user/src

COPY user/requirements.txt /app/user/requirements.txt
RUN pip install --no-cache-dir -r /app/user/requirements.txt

COPY user/src /app/user/src

EXPOSE 8080

ENTRYPOINT ["python", "/app/user/src/server.py"]
