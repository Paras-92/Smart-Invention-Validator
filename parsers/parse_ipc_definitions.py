import xml.etree.ElementTree as ET
import json

INPUT_PATH = "data/IPC_CPC_raw/ipc_fixed_texts_20250101/EN_ipc_fixed_texts_20250101.xml"
OUTPUT_PATH = "data/ipc_code_definitions.json"

def parse_ipc_fixed_texts(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    ns = {"ipc": "http://www.wipo.int/classifications/ipc/masterfiles"}
    entries = {}

    for text_node in root.findall("ipc:text", ns):
        text_id = text_node.attrib.get("id", "").strip()
        content = (text_node.text or "").strip()
        if text_id and content:
            entries[text_id] = content

    return entries

if __name__ == "__main__":
    parsed = parse_ipc_fixed_texts(INPUT_PATH)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(parsed, f, indent=2, ensure_ascii=False)

    print(f"✅ Parsed {len(parsed)} IPC fixed text entries → {OUTPUT_PATH}")
