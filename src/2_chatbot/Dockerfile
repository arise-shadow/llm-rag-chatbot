FROM python:3.9-slim

RUN pip3 install --no-cache-dir --upgrade pip 

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "app.py"]
