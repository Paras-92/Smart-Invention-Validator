import os
from serpapi import GoogleSearch
from google.cloud import bigquery

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcp_key.json"

def search_google_patents_combined(keywords):
    results = []
    for kw in keywords:
        try:
            # your scraping logic...
            # simulated placeholder:
            pat = {
                "title": f"Example patent for {kw}",
                "abstract": f"This abstract mentions {kw} multiple times.",
                "publication_date": "2020-01-01",
                "assignee": "Test Corp",
                "patent_number": f"US1234567{kw[-1]}A",
                "claim": "A dummy claim text.",
                "source": "Google Patents"
            }
            results.append(pat)
        except Exception as e:
            print(f"Error for {kw}: {e}")
            continue
    return results  # ✅ MAKE SURE THIS LINE IS INSIDE THE FUNCTION

    # --- Fallback: BigQuery ---
    try:
        client = bigquery.Client()
        keyword = keywords[0].lower()

        query = f"""
        SELECT
          publication_number,
          (SELECT title.text FROM UNNEST(title_localized) AS title LIMIT 1) AS title,
          (SELECT abstract.text FROM UNNEST(abstract_localized) AS abstract LIMIT 1) AS abstract,
          (SELECT claims.text FROM UNNEST(claims_localized) AS claims LIMIT 1) AS claim,
          filing_date,
          publication_date,
          grant_date,
          ARRAY_TO_STRING((SELECT ARRAY_AGG(name) FROM UNNEST(assignee_harmonized)), ", ") AS assignee
        FROM `patents-public-data.patents.publications`
        WHERE LOWER((SELECT abstract.text FROM UNNEST(abstract_localized) AS abstract LIMIT 1)) LIKE '%{keyword}%'
        LIMIT 5
        """
        rows = client.query(query).result()

        return [{
            "title": row.title,
            "abstract": row.abstract,
            "claim": getattr(row, "claim", ""),
            "filing_date": str(getattr(row, "filing_date", "")),
            "publication_date": str(getattr(row, "publication_date", "")),
            "grant_date": str(getattr(row, "grant_date", "")),
            "assignee": getattr(row, "assignee", ""),
            "patent_number": row.publication_number,
            "source": "BigQuery"
        } for row in rows]
    except Exception as e:
        print("❌ BigQuery error:", e)

    return []