from fastapi import FastAPI
import os

app = FastAPI()

@app.get("/")
def root():
    return {"service": "Amantia causal safety engine", "status": "running"}
