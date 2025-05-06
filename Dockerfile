FROM python:3.11

COPY src/requirements.txt /app/

WORKDIR /app

RUN pip install llama-cpp-python

RUN pip install beautifulsoup4 Flask Flask_Cors html2text llama_index Requests transformers torch python-dotenv numpy faiss-cpu llama-index-vector-stores-faiss

RUN pip install llama-index-llms-llama-cpp llama-index-embeddings-huggingface llama-index-llms-huggingface

RUN pip install wikibaseintegrator

RUN pip install llama-index-vector-stores-elasticsearch

RUN pip install flask[async] nest_asyncio

RUN pip install llama-index-llms-ollama

#RUN pip install accelerate

#RUN pip install -r requirements.txt
