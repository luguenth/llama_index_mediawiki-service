import os
import time
from llama_index.core import Settings, StorageContext, VectorStoreIndex, load_index_from_storage

import llama_index
from Models import Models
from DocumentClass import DocumentClass
#import faiss
from llama_index.core import PropertyGraphIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from WikibasePropertyGraph import WikibasePropertyGraphStore
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
import json
from llama_index.core.indices.property_graph import (
            VectorContextRetriever,
            LLMSynonymRetriever,
        )

from llama_index.core.prompts import RichPromptTemplate

class MediawikiLLM:
    MODEL_PROVIDER_OLLAMA = "ollama"
    MODEL_PROVIDER_GWDG_SAIA = "gwdg_saia"
    mediawiki_url = None
    api_url = None

    DocumentClass = None
    mw_index = None # mediawiki article content
    wb_index = None # wikibase graph 
    index_filename = None
    query_engine = None

    def __init__(self, mediawiki_url, api_url):
        import logging
        import sys
        
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
        logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
        self.mediawiki_url = mediawiki_url
        self.mediawiki_api_url = api_url
        self.DocumentClass = DocumentClass(api_url)

        #llm = Models.CreateLlamaCCP(model_url=os.getenv("MODEL_URL"), model_path=os.getenv("MODEL_PATH"))

        #llm = Models.CreateHuggingFaceLLM(model_name="TheBloke/em_german_7b_v01-GGUF")
        #llm = Models.CreateHuggingFaceLLM(model_name="meta-llama/Llama-2-7b-chat-hf")


        #llm = HuggingFaceLLM(
        #    model_name="Qwen/Qwen3-1.7B",
        #    tokenizer_name="Qwen/Qwen3-1.7B",
        #    context_window=4096,
        #    max_new_tokens=128,
            #generate_kwargs={"temperature": 0.7, "top_k": 50, "top_p": 0.95},
            #tokenizer_kwargs={"enable_thinking":False},
        #    generate_kwargs={"n_threads": 16, "use_fp8": True},
        #    messages_to_prompt=self.messages_to_prompt,
        #    completion_to_prompt=self.completion_to_prompt,
        #    device_map="auto",
        #)

        #llm = Models.CreateAutoModelForCausalLM(model_name=os.getenv("MODEL_PATH"), model_path=os.getenv("MODEL_PATH"))
        #llm = llm = AutoModelForCausalLM.from_pretrained("TheBloke/em_german_7b_v01-GGUF", model_file="em_german_7b_v01.Q2_K.gguf", model_type="llama", gpu_layers=0)

        #from llama_index.llms.llama_cpp import LlamaCPP

        #query_wrapper_prompt = RichPromptTemplate("Du bist ein hilfreicher Assistent. USER: {query_str} ASSISTANT:") 
        #llm = LlamaCPP(
        #    model_path=os.getenv("MODEL_PATH"),
        #    temperature=0.0,
        #    max_new_tokens=128,
        #    context_window=4096,
        #    generate_kwargs={},
        #    model_kwargs={"n_gpu_layers": 0, "n_threads": 16, "use_fp8": True},
        #    verbose=True,
        #)

        print("Model provider: "+os.getenv("MODEL_PROVIDER"))
        print("Model name: "+ os.getenv("MODEL_NAME"))
        print("Ollama url: "+os.getenv("OLLAMA_URL"))
        self.model_provider = os.getenv("MODEL_PROVIDER").lower()
        if not self.model_provider or self.model_provider == self.MODEL_PROVIDER_OLLAMA:
            Settings.llm = Models.CreateOllamaModel(
                model_name=os.getenv("MODEL_NAME"),
                timeout=120.0,
                url=os.getenv("OLLAMA_URL"),
                parameter={"temperature":0, "num_thread": 16}
            )  
        elif self.model_provider == self.MODEL_PROVIDER_GWDG_SAIA:
            Settings.llm = Models.callCreateOpenAIModel(
                model_name=os.getenv("MODEL_NAME"),
                api_key = os.getenv("GWDG_SAIA_KEY"),
                api_url = os.getenv("GWDG_SAIA_URL"),
                max_completion_tokens=os.getenv("GWDG_SAIA_MAXTOKENS"),
            )
        else:
            Settings.llm = None
        
        #Settings.embed_model = "local"
        #Settings.model_name = "em_german_7b_v01.Q2_K.gguf"
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        Settings.chunk_size=4096
    
    def completion_to_prompt(self, completion):
        return f"<|im_start|>system\n<|im_end|>\n<|im_start|>user\n{completion}<|im_end|>\n<|im_start|>assistant\n"

    def messages_to_prompt(self, messages):
        prompt = ""
        for message in messages:
            if message.role == "system":
                prompt += f"<|im_start|>system\n{message.content}<|im_end|>\n"
            elif message.role == "user":
                prompt += f"<|im_start|>user\n{message.content}<|im_end|>\n"
            elif message.role == "assistant":
                prompt += f"<|im_start|>assistant\n{message.content}<|im_end|>\n"

        if not prompt.startswith("<|im_start|>system"):
            prompt = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n" + prompt

        prompt = prompt + "<|im_start|>assistant\n"

        return prompt

    def init_from_mediawiki(self):
        #set_global_service_context(self.service_context)

        #d = 1536
        #faiss_index = faiss.IndexFlatL2(d)
        #vector_store = FaissVectorStore(faiss_index=faiss_index)
        vector_store = ElasticsearchStore(
            es_url="http://elasticsearch-llm:9200",#os.getenv("ELASTICSEARCH_URL"),  # see Elasticsearch Vector Store for more authentication options
            es_user="elastic",
            es_password="changeme",
            index_name="wiki-llm-vectorstore-mediawikipages",
            use_async=False
        )

        if os.path.isdir(str(os.getenv("PERSISTENT_STORAGE_DIR"))):
            storage_context = StorageContext.from_defaults(
                persist_dir=os.getenv("PERSISTENT_STORAGE_DIR"), vector_store=vector_store)
            self.mw_index = load_index_from_storage(storage_context)
        else:
            self.DocumentClass.mediawiki_get_all_pages(self.mediawiki_api_url)

            self.mw_index = VectorStoreIndex.from_documents(
                self.DocumentClass.documents)
            if os.getenv("PERSISTENT_STORAGE_DIR") is not None:
                self.mw_index.storage_context.persist(
                    os.getenv("PERSISTENT_STORAGE_DIR"))

        self.query_engine = self.mw_index.as_query_engine()

    def init_from_wikibase(self):

        #d = 384
        #d = 768
        #faiss_index = faiss.IndexFlatL2(d)
        
        vector_store = ElasticsearchStore(
            es_url="http://elasticsearch-llm:9200",#os.getenv("ELASTICSEARCH_URL"),  # see Elasticsearch Vector Store for more authentication options
            es_user="elastic",
            es_password="changeme",
            index_name="wiki-llm-vectorstore",
            use_async=False
        )
        #vector_store = FaissVectorStore(faiss_index=faiss_index)
        store_file = "propertyGraphStoreDump.json"
        if os.path.exists(store_file):
            print("load store from file...")
            with open(store_file, "r") as f:
                data = json.load(f)
                graph_store = WikibasePropertyGraphStore.from_dict(
                    data=data,
                    embed_model=Settings.embed_model,
                    vector_store=vector_store,
                )
            
            print(str(len(graph_store.get()))+" graph nodes loaded.")
        else:
            graph_store = WikibasePropertyGraphStore(
                embed_model=Settings.embed_model,
                vector_store=vector_store 
            )
            print("init store from wikibase...")
            graph_store.init_graph_from_wiki(
                os.getenv("MEDIAWIKI_API_URL"),
                os.getenv("MEDIAWIKI_USERNAME"),
                os.getenv("MEDIAWIKI_USERPASS"), )

            print("persist store...")
            #see https://github.com/run-llama/llama_index/issues/15822
            data = graph_store.graph.model_dump_json()#serialize_as_any=True
            with open(store_file, "w") as f:
                f.write(data)
        
        self.graph_store = graph_store
        self.vector_store = vector_store
        self.wb_index = PropertyGraphIndex.from_existing(
            property_graph_store=graph_store,
            llm=Settings.llm
        )
        #nodes = graph_store.get()
        #print( nodes )
        #self.wb_index._insert_nodes_to_vector_index( nodes )
        self.wb_vectorRetriever = VectorContextRetriever(
                self.graph_store, 
                embed_model=Settings.embed_model, 
                vector_store=self.vector_store,
                 # include source chunk text with retrieved paths
                include_text=True,
                include_properties = True,
                # the number of nodes to fetch
                similarity_top_k=1,
                # the depth of relations to follow after node retrieval
                path_depth=1,)
        
        sub_retrievers = [
            self.wb_vectorRetriever
            #LLMSynonymRetriever(graph_store),
        ]

        self.query_engine = self.wb_index.as_query_engine(sub_retrievers=sub_retrievers)

        if os.getenv("PROMPT"):
            qa_tmpl = RichPromptTemplate(os.getenv("PROMPT"))
            QA_PROMPT_KEY = "response_synthesizer:text_qa_template"
            self.query_engine.update_prompts({QA_PROMPT_KEY: qa_tmpl})


        #retriever = PGRetriever(sub_retrievers=sub_retrievers)
        #nodes = retriever.retrieve("Sidonia von Borcke")
        #print("retrieved nodes:"+str(nodes))

        #from llama_index.core.query_engine import RetrieverQueryEngine
        #from llama_index.core.retrievers import KnowledgeGraphRAGRetriever
        #storage_context = StorageContext.from_defaults(graph_store=graph_store)
        #graph_rag_retriever = KnowledgeGraphRAGRetriever(
        #    storage_context=storage_context,
        #    llm=llm,
        #    retriever_mode="keyword",
        #    graph_traversal_depth=2, #default 2
        #    max_knowledge_sequence=30, #default 30
        #    max_entities=5, #default 5
        #    verbose=True,
        #)

        #self.query_engine = RetrieverQueryEngine.from_args(
        #    graph_rag_retriever,
        #)

    def query(self, query:str, similarity_top_k=1, path_depth=1, show_thinking:bool=False):
        print(similarity_top_k)
        print(path_depth)
   
        self.wb_vectorRetriever.similarity_top_k = similarity_top_k
        self.wb_vectorRetriever.path_depth = path_depth
        #TODO: implement node retrieval only once (now its fetched two times, here and within query_engine.query())
        retrieved_nodes = self.wb_vectorRetriever.retrieve(query)

        if show_thinking:
            query = query + " /think"
        else:
            query = query + " /no_think"
        print(query)
        
        

        #self.query_engine = self.wb_index.as_query_engine(sub_retrievers=sub_retrievers)
        #if self.model_provider == self.MODEL_PROVIDER_OLLAMA:

        response = self.query_engine.query(query)
        #elif self.model_provider == self.MODEL_PROVIDER_GWDG_SAIA:
            #retrieve nodes
        #    nodes = self.wb_vectorRetriever.retrieve(query)
            #create response via SAIA
            # Get response
        #chat_completion = llm.chat(
        #    messages=[
        #        ChatMessage(role="system",content="You are a helpful assistant. use this context to answer:"+str(context)),
        #        ChatMessage(role="user",content=query)
        #    ],
        #    model= os.getenv("MODEL_NAME")
        #)
        print(response)
        result = {
            "answer": str(response),
            "source_content": [
                {
                    "text": node.node.text,
                    "score": node.score
                } for node in retrieved_nodes
            ]
        }
        return result
    
    async def aquery(self, query:str):
        response = await self.query_engine.aquery(query)
        print(response)
        return response
    
    def init_no_documents(self):
        self.mw_index = llama_index.indices.empty.EmptyIndex()
        self.query_engine = self.mw_index.as_query_engine()

    def updateVectorStore(self, type: str, page_url: str):
        if type == 'edit' or type == 'create':
            print("create/edit " + page_url)
            self.DocumentClass.mediawiki_update_page(page_url)

        elif type == 'delete':
            print("delete " + page_url)
            self.DocumentClass.mediawiki_delete_page(page_url)

        self.mw_index.refresh(self.DocumentClass.documents)
