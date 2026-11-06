from pathlib import Path
from paddleocr import PaddleOCR

_ENGINES = {}  # نحمّل كل موديل مرة واحدة بس
_LANG_MAP = {"en": "en", "ar": "ar"}
SEP_GAP = 1.5  # فجوة أكبر من 1.5 x ارتفاع السطر = فاصل أعمدة (يتكتب tab)


def _get_engine(lang: str, use_gpu: bool = True):
    key = (lang, use_gpu)
    if key not in _ENGINES:
        _ENGINES[key] = PaddleOCR(
            lang=_LANG_MAP[lang],
            device="gpu" if use_gpu else "cpu",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
    return _ENGINES[key]


def _run(image_path: str, lang: str, use_gpu: bool):
    engine = _get_engine(lang, use_gpu)
    res = engine.predict(image_path)[0]

    items = []
    for text, score, box in zip(res["rec_texts"], res["rec_scores"], res["rec_boxes"]):
        x1, y1, x2, y2 = [float(v) for v in box]
        items.append({"x1": x1, "x2": x2, "x": (x1 + x2) / 2, "yc": (y1 + y2) / 2,
                      "h": max(y2 - y1, 1.0), "text": text, "score": float(score)})
    if not items:
        return "", 0.0

    # ارتفاع السطر المعتاد = وسيط ارتفاعات كل الصناديق
    hs = sorted(i["h"] for i in items)
    med_h = hs[len(hs) // 2]

    items.sort(key=lambda i: i["yc"])
    lines = []
    for it in items:
        if lines:
            mean_y = sum(x["yc"] for x in lines[-1]) / len(lines[-1])
            if abs(it["yc"] - mean_y) < 0.5 * med_h:
                lines[-1].append(it)
                continue
        lines.append([it])

    rtl = lang == "ar"
    out = []
    for line in lines:
        line.sort(key=lambda i: i["x"], reverse=rtl)  # العربي يمين لشمال
        parts = [line[0]["text"]]
        for prev, cur in zip(line, line[1:]):
            gap = (prev["x1"] - cur["x2"]) if rtl else (cur["x1"] - prev["x2"])
            parts.append("\t" if gap > SEP_GAP * med_h else " ")
            parts.append(cur["text"])
        out.append("".join(parts))

    avg = sum(i["score"] for i in items) / len(items)
    return "\n".join(out), avg


def extract_text_from_image(image_path: str, lang: str = "auto", use_gpu: bool = True) -> dict:
    """يرجّع {"text": ..., "confidence": ..., "lang": ...}"""
    if not Path(image_path).is_file():
        raise FileNotFoundError(f"File not found: {image_path}")

    if lang in _LANG_MAP:
        text, conf = _run(image_path, lang, use_gpu)
        return {"text": text, "confidence": round(conf, 3), "lang": lang}

    # auto: نجرّب الاتنين ونختار الأعلى ثقة
    results = {l: _run(image_path, l, use_gpu) for l in _LANG_MAP}
    best = max(results, key=lambda l: results[l][1])
    text, conf = results[best]
    return {"text": text, "confidence": round(conf, 3), "lang": best}