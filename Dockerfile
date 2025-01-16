FROM nvidia/cuda:12.6.3-base-ubuntu22.04

WORKDIR /app
COPY . /app

RUN apt-get update && \
    apt-get install -y git python3.10 python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN pip install -r requirements.txt

EXPOSE 8501
ENV NAME=Checkr

CMD ["streamlit", "run", "app.py"]