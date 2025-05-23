FROM python:3.11

ARG MODEL_PROVIDER
ENV MODEL_PROVIDER=${MODEL_PROVIDER}

COPY src/requirements.txt /app/

WORKDIR /app

#RUN pip install llama-cpp-python

RUN pip install beautifulsoup4 html2text llama_index Requests 

RUN pip install uvicorn fastapi

# RUN pip install torch faiss-cpu numpy llama-index-vector-stores-faiss llama-index-llms-llama-cpp llama-index-llms-huggingface transformers python-dotenv  

RUN pip install llama-index-embeddings-huggingface 

RUN pip install wikibaseintegrator

RUN pip install llama-index-vector-stores-elasticsearch

# Conditionally install extras
RUN if [ "$MODEL_PROVIDER" = "ollama" ]; then \
      echo "Installing Ollama dependencies for local model execution..."; \
      pip install llama-index-llms-ollama; \
    elif [ "$MODEL_PROVIDER" = "gwdg_saia" ]; then \
      echo "Installing GWDG-SAIA dependencies for remote model execution..."; \
      pip install llama-index-llms-openai-like; \
    fi

#RUN pip install -r requirements.txt
