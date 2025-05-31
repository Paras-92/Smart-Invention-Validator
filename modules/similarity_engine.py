import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

try:
    import tensorflow_hub as hub
    model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
except:
    model = None


def score_similarity(user_text, patent_list, top_k=5):
    """
    Compare user abstract to retrieved patents using USE embeddings.
    """
    if model is None or not user_text.strip():
        return []

    abstracts = [pat["abstract"] for pat in patent_list]
    inputs = [user_text] + abstracts
    embeddings = model(inputs)
    user_vec = embeddings[0].numpy().reshape(1, -1)
    patent_vecs = embeddings[1:].numpy()

    scores = cosine_similarity(user_vec, patent_vecs)[0]

    for i, pat in enumerate(patent_list):
        pat["similarity"] = float(scores[i])

    return sorted(patent_list, key=lambda x: x["similarity"], reverse=True)[:top_k]
