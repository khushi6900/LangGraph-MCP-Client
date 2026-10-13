import sys
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import uvicorn

if __name__ == "__main__":
    config = uvicorn.Config("app.main:app", host="127.0.0.1", port=8001, loop="none")
    server = uvicorn.Server(config)
    asyncio.run(server.serve())