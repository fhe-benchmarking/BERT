from transformers import AutoTokenizer, BertForSequenceClassification

MODEL_ID = "google-bert/bert-base-cased-finetuned-mrpc"


def get_model(device="cpu"):
    model = BertForSequenceClassification.from_pretrained(MODEL_ID).to(device)
    return model


def get_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_ID)
