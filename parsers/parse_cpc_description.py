from bs4 import BeautifulSoup
from bs4 import XMLParsedAsHTMLWarning
import warnings
import os
import json

# Silence that warning
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

def parse_flat_cpc_folder(folder_path):
    cpc_entries = {}

    for file in os.listdir(folder_path):
        if file.endswith(".xml") and "cpc-scheme-" in file:
            code = file.replace("cpc-scheme-", "").replace(".xml", "").strip()

            with open(os.path.join(folder_path, file), "r", encoding="utf-8") as f:
                soup = BeautifulSoup(f, "lxml")
                text = soup.get_text(separator=" ", strip=True)
                clean = " ".join(text.split())

                # Extract title: assume first sentence or first capital block
                title = clean[:clean.find(".")+1] if "." in clean else clean[:100]
                cpc_entries[code] = {
                    "title": title,
                    "raw_text": clean
                }

    return cpc_entries

if __name__ == "__main__":
    input_folder = "data/IPC_CPC_raw/CPCSchemeXML202505"
    output_file = "data/cpc_code_descriptions.json"

    data = parse_flat_cpc_folder(input_folder)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✅ Parsed {len(data)} CPC entries → {output_file}")
