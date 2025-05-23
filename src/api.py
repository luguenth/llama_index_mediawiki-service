from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
import asyncio
from webhookParser import WebhookParser

router = APIRouter()  # <--- This needs to be imported into app.py

class WebhookRequest(BaseModel):
    content: str

class MediawikiLLMAPI:
    def __init__(self, MediawikiLLM):
        self.MediawikiLLM = MediawikiLLM
        self.executor = ThreadPoolExecutor()
        self.register_routes()

    def register_routes(self):
        @router.get("/status")
        async def run_status_on_llm():
            return JSONResponse(status_code=200, content={"status": "ready"})

        @router.get("/query")
        async def run_query(query: str, similarity_top_k: int = 1, path_depth: int = 1, show_thinking: bool = False):
            response = await self.MediawikiLLM.aquery(query, similarity_top_k, path_depth, show_thinking)
            return JSONResponse(content=response)

        @router.get("/llm")
        async def run_query_on_llm(query: str):
            response = self.MediawikiLLM.service_context.llm.complete(query)
            return JSONResponse(content={"text": response.text})

        @router.post("/webhook")
        async def webhook(payload: WebhookRequest):
            type, page_url = WebhookParser.parse(content=payload.content)
            if not type:
                raise HTTPException(status_code=400, detail="Error in webhook")
            self.MediawikiLLM.updateVectorStore(type, page_url)
            return JSONResponse(status_code=204, content=None)
