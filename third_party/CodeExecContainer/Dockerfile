FROM python:3.9-slim

# gcc와 필요한 라이브러리 설치
RUN apt-get update && apt-get install -y \
    gcc \
    libpng-dev \
    libfreetype6-dev \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉터리 설정
WORKDIR /app

# requirements.txt 파일 복사
COPY requirements.txt /app/

# 필요한 라이브러리를 설치
RUN pip install -r requirements.txt