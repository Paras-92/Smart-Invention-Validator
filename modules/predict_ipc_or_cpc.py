# modules/predict_ipc_or_cpc.py

import joblib
import pandas as pd
import json
import os
import numpy as np

def load_descriptions(type_="ipc"):
    """Load IPC or CPC code descriptions."""
    file_map = {
        "ipc": "data/ipc_code_descriptions.json",
        "cpc": "data/cpc_code_descriptions.json"
    }
    with open(file_map[type_], "r", encoding="utf-8") as f:
        return json.load(f)

def get_definition_with_fallback(ipc_code, descriptions_dict):
    """Fallback to shorter codes (G06F → G06 → G) if full code not found."""
    if ipc_code in descriptions_dict:
        return descriptions_dict[ipc_code]

    fallback_codes = [
        ipc_code[:4],  # e.g. G06F
        ipc_code[:3],  # e.g. G06
        ipc_code[0]    # e.g. G
    ]
    for code in fallback_codes:
        if code in descriptions_dict:
            return descriptions_dict[code]
    return {}

def load_model(model_type):
    """Load a model bundle from disk."""
    model_path = f"ipc_classifier/{model_type}_model.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model not found: {model_path}")
    return joblib.load(model_path)

def predict_codes(abstract, model_type="ipc", top_k=4, custom_model=None):
    """
    Predict IPC or CPC codes with confidence scores.

    Args:
        abstract: Invention abstract (string)
        model_type: "ipc" or "cpc"
        top_k: number of top predictions to return
        custom_model: dictionary with keys "model", "code_descriptions", etc.

    Returns:
        List of dicts: [{code, title, section, confidence}, ...]
    """
    model_bundle = custom_model if custom_model else load_model(model_type)

    model = model_bundle["model"]
    descriptions = model_bundle.get("code_descriptions") or load_descriptions(model_type)

    # Predict
    probs = model.predict_proba([abstract])[0]
    classes = model.classes_
    top_indices = np.argsort(probs)[::-1][:top_k]

    results = []
    for i in top_indices:
        code = classes[i]
        score = probs[i]
        definition = get_definition_with_fallback(code, descriptions)
        title = definition.get("title", "N/A")
        section = definition.get("section", "Unknown")
        results.append({
            "code": code,
            "title": title,
            "section": section,
            "confidence": round(score * 100, 2)
        })

    return results
