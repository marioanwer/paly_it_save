import os
import re
import tempfile
import fitz  # PyMuPDF
from .image_extractor import extract_text_from_image

ARABIC_RE = re.compile(r"[؀-ۿ]")


def _render_scale(page, dpi: int, max_side: int) -> float:
    """dpi المطلوب، بس من غير ما أكبر ضلع في الصورة يعدي max_side بكسل
    (صفحات الويب الطويلة بتطلع صور ضخمة وبتاكل ذاكرة، والـ OCR بيصغّرها لـ 4000 على أي حال)."""
    scale = dpi / 72.0
    longest = max(page.rect.width, page.rect.height) * scale
    if longest > max_side:
        scale *= max_side / longest
    return scale


def extract_text_from_pdf(pdf_path: str, lang: str = "auto", use_gpu: bool = True,
                          dpi: int = 200, max_side: int = 3500) -> list:
    pages = []
    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        # النص الجاهز في PDF العربي بيطلع بحروف ناقصة وترتيب مقلوب، فنعمل OCR بدله
        arabic_text = bool(ARABIC_RE.search(text)) and lang != "en"
        if len(text) > 30 and not arabic_text:
            pages.append({"page": i, "raw_text": text, "ocr_confidence": None})
        else:  # صفحة ممسوحة أو عربي: نحولها لصورة ونعمل OCR
            s = _render_scale(page, dpi, max_side)
            pix = page.get_pixmap(matrix=fitz.Matrix(s, s))
            with tempfile.TemporaryDirectory() as d:
                img_path = os.path.join(d, "page.png")
                pix.save(img_path)
                r = extract_text_from_image(img_path, lang, use_gpu)
            pages.append({"page": i, "raw_text": r["text"], "ocr_confidence": r["confidence"]})
    doc.close()
    return pages