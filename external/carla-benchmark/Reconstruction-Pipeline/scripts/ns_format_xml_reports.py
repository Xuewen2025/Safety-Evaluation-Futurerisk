from lxml import etree
import os
from settings import REPORTS_DIR

parser = etree.XMLParser(remove_blank_text=True)

for filename in os.listdir(REPORTS_DIR):
    if filename.endswith(".xml"):
        filepath = os.path.join(REPORTS_DIR, filename)
        try:
            tree = etree.parse(filepath, parser)
            tree.write(filepath, pretty_print=True, encoding='utf-8', xml_declaration=True)
            print(f"Formatted: {filename}")
        except Exception as e:
            print(f"Error on {filename}: {e}")