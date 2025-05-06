import time
from flask import request, jsonify, stream_with_context
import json
from webhookParser import WebhookParser
import html

# wird verwendet, um es zu ermöglichen, dass ein bereits laufender asyncio-Eventloop erneut verwendet werden kann (avoid runtime error "detected nested async")
import asyncio
import nest_asyncio
from concurrent.futures import ThreadPoolExecutor

class MediawikiLLMAPI:
    def __init__(self, app,  MediawikiLLM):
        nest_asyncio.apply()
        self.MediawikiLLM = MediawikiLLM
        self.executor = ThreadPoolExecutor()

        @app.route('/query', methods=['GET'])
        async def run_query():
            query = request.args.get('query')
            #response = await MediawikiLLM.query(query)
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                self.executor, lambda: self.MediawikiLLM.query(query)
            )
            return jsonify(response.response)
            
        @app.route('/llm', methods=['GET'])
        def run_query_on_llm():
            query = request.args.get('query')
            print(query)
            response = MediawikiLLM.service_context.llm.complete(query)
            print(response)
            return jsonify(response.text)

        @app.route('/webhook', methods=['POST'])
        def webhook():
            content = json.loads(request.data)['content']
            type, page_url = WebhookParser.parse(content=content)

            if (type == ""):
                return ('error in webhook', 400)
            else:
                MediawikiLLM.updateVectorStore(type, page_url)
                return ('', 204)

        app.run(host='0.0.0.0', port=5000, debug=False)
