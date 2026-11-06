import os
import re
import tempfile
import fitz  # PyMuPDF
from .image_extractor import extract_text_from_image

ARABIC_RE = re.compile(r"[\u0600-\u06FF]")


def extract_text_from_pdf(pdf_path: str, lang: str = "auto", use_gpu: bool = True, dpi: int = 200) -> list:
    pages = []
    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        # النص الجاهز في PDF العربي بيطلع بحروف ناقصة وترتيب مقلوب، فنعمل OCR بدله
        arabic_text = bool(ARABIC_RE.search(text)) and lang != "en"
        if len(text) > 30 and not arabic_text:
            pages.append({"page": i, "raw_text": text, "ocr_confidence": None})
        else:  # صفحة ممسوحة أو عربي: نحولها لصورة ونعمل OCR
            pix = page.get_pixmap(dpi=dpi)
            with tempfile.TemporaryDirectory() as d:
                img_path = os.path.join(d, "page.png")
                pix.save(img_path)
                r = extract_text_from_image(img_path, lang, use_gpu)
            pages.append({"page": i, "raw_text": r["text"], "ocr_confidence": r["confidence"]})
    doc.close()
    return pages