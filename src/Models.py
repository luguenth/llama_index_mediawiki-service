#from llama_index.core.prompts import RichPromptTemplate
#from llama_index.llms.llama_cpp import LlamaCPP
#from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.ollama import Ollama
import requests
import os
from llama_index.llms.openai_like import OpenAILike

from llama_index.core.llms import ChatMessage
class Models():
    def CreateOllamaModel(model_name="qwen3:0.6B",timeout=120.0, url="http://ollama-llm:11434", parameter={"num_thread": 16}):
        # Pull a model (e.g., llama3, qwen3:0.6B)
        response = requests.post(url+"/api/pull", json={"name": model_name})
        # Check result
        print(response.text)
        llm = Ollama(model=model_name, request_timeout=120.0, base_url=url, additional_kwargs=parameter)
        return llm

    def callCreateOpenAIModel(model_name="qwen3-32b", api_url=None, api_key=None, max_completion_tokens=None):
       
        # Start OpenAI client
        llm = OpenAILike(
            model=os.getenv("MODEL_NAME"),
            api_key = os.getenv("GWDG_SAIA_KEY"),
            api_base = os.getenv("GWDG_SAIA_URL"),
            max_tokens=max_completion_tokens
        )
        return llm
        
    #def CreateLlamaCCP(model_url: str = None, model_path: str = None):
    #    llm = LlamaCPP(
    #        model_url=model_url,
    #        model_path=model_path,
    #        temperature=0.1,
    #        max_new_tokens=256,
    #        context_window=3900,
    #        generate_kwargs={},
    #        model_kwargs={"n_gpu_layers": 41, "n_threads": 16},
    #        messages_to_prompt=messages_to_prompt,
    #        completion_to_prompt=completion_to_prompt,
    #        verbose=True,
    #    )
    #    return llm

    #def CreateHuggingFaceLLM(model_name: str):

    #    query_wrapper_prompt = RichPromptTemplate(
    #        "Below is an instruction that describes a task. "
    #        "Write a response that appropriately completes the request.\n\n"
    #        "### Instruction:\n{query_str}\n\n### Response:"
    #    )

    #    llm = HuggingFaceLLM(
    #        context_window=512,
    #        max_new_tokens=256,
    #        generate_kwargs={"temperature": 0.25, "do_sample": False},
    #        query_wrapper_prompt=query_wrapper_prompt,
    #        tokenizer_name=model_name,
    #        model_name=model_name,
    #        device_map="auto",
    #        tokenizer_kwargs={"max_length": 2048}
    #    )
    #    return llm

    #def CreateAutoModelForCausalLM(model_name: str = None, model_path: str = None, model_type: str = None):
    #    llm = AutoModelForCausalLM.from_pretrained(model_name, model_file=model_path, model_type="llama", gpu_layers=0)
    #    return  llm

    def RemoteHF(model_name: str):
        llm = HuggingFaceInferenceAPI(model_name=model_name,token="yourtoken")
        return llm

