from fastapi import FastAPI

from .routers import chat, config, model_verify, openai, tools

app = FastAPI()

app.include_router(openai)
app.include_router(chat)
app.include_router(tools)
app.include_router(config)
app.include_router(model_verify)


# memo: fastapi dev app.py
