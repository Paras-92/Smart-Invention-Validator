# ipc_classifier/train_ipc_model.py

import pandas as pd
import joblib
import json
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

# Load IPC descriptions
with open("data/ipc_code_descriptions.json", "r", encoding="utf-8") as f:
    ipc_descriptions = json.load(f)

def train_and_save_ipc_model():
    print("📄 Loading IPC data...")
    df = pd.read_csv("data/ipc_data.csv")
    df.dropna(subset=["abstract", "ipc_codes"], inplace=True)

    # Optional: use only a subset to speed up testing
    df = df.sample(8000, random_state=42)

    tqdm.pandas()  # Activate progress bar

    print("🧠 Enriching text with descriptions...")
    df["text"] = df.progress_apply(
        lambda row: f"{row['abstract']} {ipc_descriptions.get(row['ipc_codes'].strip(), {}).get('title', '')}",
        axis=1
    )

    texts = df["text"].tolist()
    labels = df["ipc_codes"].str.strip().tolist()

    print(f"🌲 Training Logistic Regression on {len(texts)} samples...")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=2000)),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(texts, labels)

    model_bundle = {
    "model": pipeline,
    "vectorizer": pipeline.named_steps["tfidf"],
    "code_descriptions": ipc_descriptions,
    "code_definitions": {}
    }
    joblib.dump(model_bundle, "ipc_classifier/ipc_model.pkl")

    print("✅ IPC model saved to ipc_classifier/ipc_model.pkl")

if __name__ == "__main__":
    train_and_save_ipc_model()