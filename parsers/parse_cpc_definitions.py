    # parse_cpc_definitions_folder.py

import os
import json
from bs4 import BeautifulSoup
from bs4 import XMLParsedAsHTMLWarning
import warnings

# Suppress XMLParsedAsHTMLWarning
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

def parse_cpc_definition_folder(folder_path):
    definitions = {}

    for file in os.listdir(folder_path):
        if file.endswith(".xml") and "cpc-definition-" in file:
            code = file.replace("cpc-definition-", "").replace(".xml", "").strip()

            with open(os.path.join(folder_path, file), "r", encoding="utf-8") as f:
                soup = BeautifulSoup(f, "lxml")
                text = soup.get_text(separator=" ", strip=True)
                clean = " ".join(text.split())
                definitions[code] = clean

    return definitions

if __name__ == "__main__":
    input_folder = "data/IPC_CPC_raw/FullCPCDefinitionXML202505"
    output_file = "data/cpc_code_definitions.json"

    parsed_defs = parse_cpc_definition_folder(input_folder)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(parsed_defs, f, indent=2, ensure_ascii=False)

    print(f"✅ Parsed {len(parsed_defs)} CPC definitions → {output_file}")
