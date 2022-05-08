from fastapi import FastAPI, responses, Query
import logging
from transformers import pipeline, TFAutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger("my-project-logger")
#load the model
model_path = "app/ml_models/distilbert-base-uncased-finetuned-sst-2-english"
model = TFAutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True, from_pt=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
sentiment_model = pipeline("sentiment-analysis",  model=model, tokenizer=tokenizer)
print("----------- transformer model loaded ------------")

#start app
app = FastAPI(docs_url='/')

@app.get("/health")
def read_root():
    return {"status": "ok"}

@app.get("/prediction/sentiment/")
def sentiment_prediction(sentence: str = Query("999", min_length=2, max_length=50)):
    try:
        number = int(sentence)
        if isinstance(number, int):
            raise ValueError
        number = float(sentence)
        if isinstance(number, float):
            raise ValueError
    except ValueError as ve:
        return responses.JSONResponse(
            content={"error": 'You cannot analyze sentiment of numbers'},
            status_code=400
        )

    if sentence is None:
        return responses.JSONResponse(
            content={"error": "You must to introduce a sentence"},
            status_code=400
        )

    result = sentiment_model(sentence)
    return {"label": result[0]['label'], "score": result[0]['score']}
