FROM python:3.12-slim
RUN pip install --no-cache-dir pytest \
    && mkdir /sandbox /workspace \
    && chmod 1777 /sandbox /workspace
WORKDIR /workspace
