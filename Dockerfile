FROM python:3.11

COPY src/requirements.txt /app/

WORKDIR /app

#RUN pip install llama-cpp-python

RUN pip install beautifulsoup4 Flask Flask_Cors html2text llama_index Requests 

# RUN pip install torch faiss-cpu numpy llama-index-vector-stores-faiss llama-index-llms-llama-cpp llama-index-llms-huggingface transformers python-dotenv  

RUN pip install llama-index-embeddings-huggingface 

RUN pip install wikibaseintegrator

RUN pip install llama-index-vector-stores-elasticsearch

RUN pip install flask[async] asyncio nest_asyncio

RUN pip install llama-index-llms-ollama

#RUN pip install llama-index-llms-openai

RUN pip install llama-index-llms-openai-like

#RUN pip install accelerate

#RUN pip install -r requirements.txt
