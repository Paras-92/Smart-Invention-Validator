# modules/keyword_extractor.py

import yake

def extract_keywords(text: str, max_keywords: int = 10) -> list:
    """
    Extract keywords using YAKE — fast, lightweight, and reliable.

    Args:
        text (str): The input abstract text.
        max_keywords (int): Number of keywords to extract.

    Returns:
        List[str]: Extracted keywords.
    """
    if not text.strip():
        return []

    text = text.strip()
    extractor = yake.KeywordExtractor(top=max_keywords, stopwords=None)
    raw_keywords = extractor.extract_keywords(text)
    return [kw[0] for kw in raw_keywords]
