FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir \
    --index-url https://pypi.org/simple/ \
    --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    -r requirements.txt

COPY . .

CMD ["python", "detect.py"]

