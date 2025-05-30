import fitz  # PyMuPDF
import docx
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Setup CUDA jika tersedia
device = 0 if torch.cuda.is_available() else -1

# Load model NER Bahasa Indonesia
model_name = "cahya/bert-base-indonesian-NER"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

ner_pipeline = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple",
    device=device
)

def extract_text(file_path):
    if file_path.endswith(".pdf"):
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    elif file_path.endswith(".docx"):
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    else:
        return "Format file tidak didukung."

def run_ner(text):
    results = ner_pipeline(text)
    df = pd.DataFrame(results)
    df = df.rename(columns={"word": "Text", "entity_group": "Entity", "score": "Confidence"})
    df["Confidence"] = df["Confidence"].round(3)
    return df

def filter_entities(df, entity_types=None, min_confidence=0.0):
    if entity_types:
        df = df[df["Entity"].isin(entity_types)]
    df = df[df["Confidence"] >= min_confidence]
    return df
