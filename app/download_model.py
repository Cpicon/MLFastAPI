from transformers import pipeline

def download_model():
    model_path = "app/ml_models/distilbert-base-uncased-finetuned-sst-2-english"
    classifier = pipeline('sentiment-analysis')
    classifier.save_pretrained(model_path)
    # %% test if it works
    print(classifier(["good"]))

if __name__ == "__main__":

    download_model()

