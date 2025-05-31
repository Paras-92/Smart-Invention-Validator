import pandas as pd
import joblib
import json
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

# Load CPC descriptions
with open("data/cpc_code_descriptions.json", "r", encoding="utf-8") as f:
    cpc_descriptions = json.load(f)

def get_hierarchy_parts(code):
    """Split CPC code into logical levels for fallback-aware learning"""
    if " " in code:
        section, rest = code.split(" ", 1)
    else:
        section, rest = code[0], code[1:]

    parts = rest.split("/")
    group = parts[0]
    subgroup = parts[-1]

    return list(set([
        f"{section}",
        f"{section} {group}",
        f"{section} {group}/{subgroup}",
        code.strip()
    ]))

def train_and_save_cpc_model():
    print("📄 Loading CPC data...")
    df = pd.read_csv("data/cpc_data.csv")
    df.dropna(subset=["abstract", "cpc_code"], inplace=True)

    # Optional: reduce dataset size for faster training
    # df = df.sample(3000, random_state=42)

    tqdm.pandas()

    texts = []
    labels = []

    print("🧠 Building training samples with hierarchy enrichment...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        abstract = row["abstract"]
        code = row["cpc_code"].strip()
        hierarchy = get_hierarchy_parts(code)

        for label in hierarchy:
            desc = cpc_descriptions.get(label, {}).get("title", "")
            full_text = f"{abstract} {desc}"
            texts.append(full_text)
            labels.append(label)

    print(f"🌲 Training Random Forest on {len(texts)} samples...")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000)),
        ("clf", RandomForestClassifier(n_estimators=50, random_state=42))
    ])

    pipeline.fit(texts, labels)

    model_bundle = {
    "model": pipeline,
    "vectorizer": pipeline.named_steps["tfidf"],
    "code_descriptions": cpc_descriptions,
    "code_definitions": {}
    }
    joblib.dump(model_bundle, "ipc_classifier/cpc_model.pkl")

    print("✅ CPC model saved to ipc_classifier/cpc_model.pkl")

if __name__ == "__main__":
    train_and_save_cpc_model()
