import json
import sys
from pathlib import Path
from src.ingestion.image_extractor import extract_text_from_image
from src.ingestion.pdf_extractor import extract_text_from_pdf
from src.extraction.structurer import structure_document

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def run_pipeline(file_path: str, lang: str = "auto", output_path: str = None, use_gpu: bool = True) -> dict:
    p = Path(file_path)
    result = {"status": "success", "source_file": p.name, "pages": [], "questions": [], "errors": []}

    # 1) قراءة الملف (OCR أو نص مباشر)
    try:
        ext = p.suffix.lower()
        if ext in IMAGE_EXTS:
            r = extract_text_from_image(str(p), lang, use_gpu)
            result["pages"].append({"page": 1, "raw_text": r["text"], "ocr_confidence": r["confidence"]})
        elif ext == ".pdf":
            result["pages"] = extract_text_from_pdf(str(p), lang, use_gpu)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    except Exception as e:
        result["status"] = "failed"
        result["errors"].append({"file": p.name, "stage": "extraction", "message": str(e)})

    # 2) استخراج الأسئلة من نص كل الصفحات مع بعض
    if result["pages"]:
        try:
            result["questions"] = structure_document(result["pages"])
            if not result["questions"]:
                result["errors"].append({"file": p.name, "stage": "structuring",
                                         "message": "no multiple-choice questions found"})
        except Exception as e:
            result["status"] = "partial"
            result["errors"].append({"file": p.name, "stage": "structuring", "message": str(e)})

    out = Path(output_path) if output_path else Path("output") / (p.stem + ".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {out} | questions: {len(result['questions'])} | errors: {len(result['errors'])}")
    return result


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tests/samples")
    lang = sys.argv[2] if len(sys.argv) > 2 else "auto"
    if target.is_dir():
        files = [f for f in sorted(target.iterdir()) if f.suffix.lower() in IMAGE_EXTS | {".pdf"}]
    else:
        files = [target]
    for f in files:
        print(f"Processing: {f.name}")
        run_pipeline(str(f), lang)