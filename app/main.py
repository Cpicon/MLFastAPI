from fastapi import FastAPI
import logging
from transformers import pipeline, TFAutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger("my-project-logger")
#load the model
model_path = "app/models/distilbert-base-uncased-finetuned-sst-2-english"
model = TFAutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True, from_pt=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
sentiment_model = pipeline("sentiment-analysis",  model=model, tokenizer=tokenizer)
print("----------- transformer model loaded ------------")

#start app
app = FastAPI(docs_url='/docs')

@app.get("/health")
def read_root():
    return {"status": "ok"}

@app.get("/prediction/sentiment/{sentence}")
def sentiment_prediction(sentence: str):
    result = sentiment_model(sentence)
    print(result)
    return {"label": result[0]['label'], "score": result[0]['score']}
