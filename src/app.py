import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from api import MediawikiLLMAPI, router
from MediawikiLLM import MediawikiLLM
from logger import logger

load_dotenv()

logger.debug("Starting MediawikiLLM initialization...")

MWLLM = MediawikiLLM(
    os.getenv("MEDIAWIKI_URL"),
    os.getenv("MEDIAWIKI_API_URL")
)

logger.debug("MediawikiLLM instance created.")
MWLLM.init_from_wikibase()
logger.debug("MediawikiLLM initialized from Wikibase.")

# ✅ Create FastAPI app instance
app = FastAPI()

# ✅ Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Pass the app to register routes
MediawikiLLMAPI(MWLLM)
app.include_router(router)  # ✅ Register the routes with FastAPI

logger.debug("FastAPI app initialized and routes set.")

if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server...")
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)
