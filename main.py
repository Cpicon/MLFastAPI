from fastapi import FastAPI
import logging
from transformers import pipeline

logger = logging.getLogger("my-project-logger")

app = FastAPI(docs_url='/docs')
sentiment_model = pipeline("sentiment-analysis")


@app.get("/health")
def read_root():
    return {"status": "ok"}

@app.get("/prediction/sentiment/{sentence}")
def sentiment_prediction(sentence: str):
    result = sentiment_model(sentence)
    print(result)
    return {"label": result[0]['label'], "score": result[0]['score']}
