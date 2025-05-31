import xml.etree.ElementTree as ET
import json

INPUT_PATH = "data/IPC_CPC_raw/ipc_scheme_20250101/EN_ipc_scheme_20250101.xml"
OUTPUT_PATH = "data/ipc_code_descriptions.json"

def get_namespace(tag):
    return tag[1:].split("}")[0] if tag.startswith("{") else ""

def parse_ipc_scheme(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ns = {'ipc': get_namespace(root.tag)}

    ipc_codes = {}

    for entry in root.findall(".//ipc:ipcEntry", ns):
        code = entry.attrib.get("symbol")
        title = ""

        # Try to find the title text
        title_tag = entry.find(".//ipc:title", ns)
        if title_tag is not None and title_tag.text:
            title = title_tag.text.strip()
        else:
            # Try fallback: first <text> element under <titlePart>
            text_elem = entry.find(".//ipc:titlePart/ipc:text", ns)
            if text_elem is not None and text_elem.text:
                title = text_elem.text.strip()

        if code and title:
            ipc_codes[code] = {
                "title": title,
                "section": code[0]
            }

    return ipc_codes

if __name__ == "__main__":
    parsed = parse_ipc_scheme(INPUT_PATH)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(parsed, f, indent=2, ensure_ascii=False)

    print(f"✅ Parsed {len(parsed)} IPC entries → {OUTPUT_PATH}")
