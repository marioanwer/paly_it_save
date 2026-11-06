import json
import sys
from pathlib import Path
from src.extraction.structurer import structure_document


def restructure(json_path: str):
    p = Path(json_path)
    data = json.loads(p.read_text(encoding="utf-8"))
    data["errors"] = [e for e in data["errors"] if e.get("stage") != "structuring"]
    data["questions"] = structure_document(data["pages"])
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Updated: {p} | questions: {len(data['questions'])}")


if __name__ == "__main__":
    restructure(sys.argv[1])