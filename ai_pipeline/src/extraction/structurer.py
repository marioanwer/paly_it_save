import re

ARABIC_RE = re.compile(r"[ء-ي]")
LABEL_RE = re.compile(r"(?<!\w)\(?([A-Da-d])[\.\)]\s*")
MARKS_RE = re.compile(
    r"^[\W_]*(?:\d+\s*)?(?:درجتان|درجتين|درجة|درجه|درجات|marks?)[\W_]*$", re.IGNORECASE)

AR_LABELS = "أبجد"
AR_PUNCT = "\\)\\.\\-\\s\"'“”"
NUM_AR_RE = re.compile(r"^\s*[\d٠-٩]{1,3}\s*[\.\)\-،:]?(?:\s+|$|(?=[ء-ي]))")

# سطر بيبدأ بعنوان مفتاح الإجابات (إنجليزي أو عربي)
KEY_MARK_RE = re.compile(
    r"(?im)^[ \t]*[^\w\n]*(?:"
    r"(?:answer\s*keys?|model\s*answers?|correct\s*answers?|answers?)"
    r"|(?:الإجابات|الاجابات|الإجابة\s*النموذجية|الاجابة\s*النموذجية|مفتاح)(?:[ \t]+[ء-ي]+){0,2}"
    r")[ \t]*[:\-–]*[ \t]*(?=\n|$|\d|[A-Da-dأبجد][ \t]*[\.\)\-])")
AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
KEY_NUM_FIRST = re.compile(
    r"(?<![\wء-ي])(\d{1,3})[ \t]*[\.\)\-–:]*[ \t]*\(?([A-Da-dأبجد])(?![A-Za-zء-ي])")
KEY_LETTER_FIRST = re.compile(
    r"(?<![\wء-ي])([A-Da-dأبجد])[ \t]*[\.\)\-–:]+[ \t]*(\d{1,3})(?!\d)")


def _ar_label_re(letter: str):
    """الحرف بعده مش حرف عربي، يعني فعلاً label مش بداية كلمة (زي 'أي')."""
    return re.compile(r"^\(?" + letter + r"(?![ء-ي])[" + AR_PUNCT + r"]*")


AR_A_RE = _ar_label_re("أ")


# ---------- أدوات عامة ----------
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def _strip_noise(q: str) -> str:
    """يشيل كلمة صغيرة مقطوعة (من الـ watermark) بعد نهاية الجملة، زي 'meadow. ses'."""
    return re.sub(r"([\.\?\!؟])\s+[a-z]{2,4}$", r"\1", q.strip())


def _clean_ocr(text: str) -> str:
    """يشيل خانات الدرجات، وفي الأوراق العربي يشيل خانات الزبالة. الإنجليزي ما بيتلمسش."""
    lines = []
    for l in text.splitlines():
        cells = [c.strip() for c in l.split("\t")
                 if c.strip() and not MARKS_RE.match(c.strip())]
        if cells:
            lines.append("\t".join(cells))
    if not lines:
        return ""

    arabic_lines = sum(1 for l in lines if ARABIC_RE.search(l))
    if arabic_lines < 0.3 * len(lines):
        return "\n".join(lines)  # ورقة إنجليزي

    out = []
    for l in lines:
        cells = [c for c in l.split("\t")
                 if ARABIC_RE.search(c) or re.fullmatch(r"[\d٠-٩\s\.\)\-]+", c)]
        if cells:
            out.append("\t".join(cells))
    return "\n".join(out)


def _build(question, choices, answer, page, ocr_conf, raw_text, has_key) -> dict:
    question = _strip_noise(re.sub(r"\s+", " ", str(question)).strip())
    choices = [re.sub(r"\s+", " ", str(c)).strip() for c in choices]
    choices = [c for c in choices if c]
    answer = re.sub(r"\s+", " ", str(answer)).strip()
    issues = []

    if not question:
        issues.append("empty_question")
    elif len(re.sub(r"[^A-Za-zء-ي]", "", question)) < 6:
        issues.append("short_question")          # زي "6." أو "A" أو "ses"
    if len(choices) != 4 or len(set(choices)) != 4:
        issues.append("bad_choices")
    if ocr_conf is not None and ocr_conf < 0.80:
        issues.append("low_ocr_confidence")

    raw_norm = _norm(raw_text)
    if any(_norm(c) not in raw_norm for c in choices):
        issues.append("choices_not_in_text")

    # الإجابة بتيجي من المفتاح بس (بيتطبق في _apply_key)
    if not has_key or answer not in choices:
        answer = ""
        issues.append("no_answer")

    return {
        "question": question,
        "choices": choices,
        "correct_answer": answer,
        "page": page,
        "confidence": ocr_conf,
        "needs_review": bool(issues),
        "issues": issues,
    }


def _failed(block: str, page: int, ocr_conf) -> dict:
    """بلوك ماقدرناش نفهمه: نسجّله للمراجعة بدل ما نخفيه."""
    return {
        "question": re.sub(r"\s+", " ", block).strip(),
        "choices": [],
        "correct_answer": "",
        "page": page,
        "confidence": ocr_conf,
        "needs_review": True,
        "issues": ["parse_failed"],
    }


def _flag_duplicates(qs: list) -> list:
    seen = set()
    for q in qs:
        key = _norm(q["question"])
        if key and key in seen:
            q["issues"].append("duplicate_question")
            q["needs_review"] = True
        seen.add(key)
    return qs


# ---------- مفتاح الإجابات ----------
def _split_key(raw_text: str):
    """يفصل نص الأسئلة عن قسم المفتاح (لو فيه). يرجّع (نص الأسئلة، نص المفتاح أو '').
    العنوان لازم يكون سطر عنوان فعلاً (مش جملة زي 'Answer the following questions')،
    والقسم لازم يتقرا منه إجابات، وإلا الورقة كلها أسئلة."""
    for m in reversed(list(KEY_MARK_RE.finditer(raw_text))):
        key_part = raw_text[m.start():]
        if len(_parse_key(key_part)) >= 2:
            return raw_text[:m.start()], key_part
    return raw_text, ""


def _parse_key(key_text: str) -> dict:
    """{رقم السؤال: رقم الاختيار 0-3}. بيقرأ 'رقم ثم حرف' (1-B) و'حرف ثم رقم' (ب-1) مع بعض."""
    text = key_text.translate(AR_DIGITS)
    if ARABIC_RE.search(text):
        # أخطاء OCR الشائعة في حروف العربي: چ بدل ج، وأ بتتقرا I أو l أو ا
        text = text.translate(str.maketrans({"چ": "ج", "إ": "أ", "آ": "أ", "ا": "أ", "I": "أ", "l": "أ", "|": "أ"}))
    key = {}
    for n, l in KEY_NUM_FIRST.findall(text):
        key.setdefault(int(n), l)
    for l, n in KEY_LETTER_FIRST.findall(text):
        key.setdefault(int(n), l)
    out = {}
    for n, l in key.items():
        l = l.upper() if l.isascii() else l
        idx = "ABCD".find(l) if l.isascii() else AR_LABELS.find(l)
        if idx >= 0:
            out[n] = idx
    return out


def _apply_key(qs: list, key: dict) -> list:
    """بيطبّق المفتاح بالترتيب: السؤال رقم i هو i-ام سؤال. لو العدد مش متطابق ما بنطبقش حاجة."""
    if not key:
        return qs
    if len(qs) != max(key):
        for q in qs:
            if "answer_key_mismatch" not in q["issues"]:
                q["issues"].append("answer_key_mismatch")
            q["needs_review"] = True
        return qs
    for i, q in enumerate(qs, start=1):
        idx = key.get(i)
        if idx is None or len(q["choices"]) != 4:
            continue
        q["correct_answer"] = q["choices"][idx]
        q["issues"] = [x for x in q["issues"] if x != "no_answer"]
        q["needs_review"] = bool(q["issues"])
    return qs


# ---------- أوراق بحروف A-D: قراءة متسلسلة (بتتحمل الاختيارات المبعثرة) ----------
JUNK_CELL_RE = re.compile(
    r"^(?:good\s*luck|d\s*luck|best\s*wishes|page\s*\d+\s*of\s*\d+.*|\d+\s*degrees?|"
    r"\w{0,10}\s*no\.?\s*\d.*|\W*end\s*of\s*question\W*)$", re.IGNORECASE)
STEM_NUM_RE = re.compile(r"^(?:[A-Za-z]{1,6}\s+)?\d{1,3}\s*[\.\)]\s*")   # 5) أو 'Exa 15)'
STEM_LOST_RE = re.compile(r"^(?:\.{1,3}|\))\s*(?=\S)")                    # '.. text' أو ')text' (الرقم ضاع)


def _tokens(line: str) -> list:
    """يقسّم السطر لخانات (tab)، وكل خانة لنص واختيارات: ('text', s) أو ('item', حرف, نص)."""
    cells = [c.strip() for c in line.split("\t") if c.strip()]
    merged, i = [], 0
    while i < len(cells):
        c = cells[i]
        if re.fullmatch(r"\d{1,3}\s*[\.\)]", c) and i + 1 < len(cells):  # '9)' لوحدها + نص السؤال
            c, i = c + " " + cells[i + 1], i + 1
        merged.append(c)
        i += 1
    out = []
    for c in merged:
        ms = list(LABEL_RE.finditer(c))
        pre = c[:ms[0].start()].strip() if ms else c
        if pre:
            out.append(("text", pre))
        for k, m in enumerate(ms):
            end = ms[k + 1].start() if k + 1 < len(ms) else len(c)
            out.append(("item", m.group(1).upper(), c[m.end():end].strip()))
    return out


def _parse_stream(text: str) -> list:
    """بيمشي على النص بالترتيب: كل 4 اختيارات = سؤال. سؤال اتلخبط ترتيبه (نصه جه قبل آخر اختيارات اللي قبله)
    بيتحط في طابور. أي كلام من غير اختيارات (مقالي، كود، هيدر) بيتتجاهل.
    يرجّع [(نص السؤال، [الاختيارات])]."""
    out, stem, items, queue = [], [], [], []

    def flush():
        nonlocal stem, items
        stray = len(items) == 1 and items[0][0] != "A"   # حرف تايه زي 'B. Using the HTML...'
        if items and not stray:
            if len(items) == 4 and sorted(l for l, _ in items) == list("ABCD"):
                items = sorted(items, key=lambda x: x[0])  # الترتيب اتلخبط في الصفحة (c قبل b)
            out.append((" ".join(stem).strip(), [t for _, t in items]))
        stem = [queue.pop(0)] if queue else []
        items = []

    for line in text.splitlines():
        for tok in _tokens(line):
            if tok[0] == "item":
                _, letter, t = tok
                if len(items) >= 4 or (letter == "A" and items):
                    flush()
                items.append((letter, t))
                continue
            s = tok[1]
            if JUNK_CELL_RE.match(s) or (len(re.sub(r"[\W_]", "", s)) <= 2 and not STEM_NUM_RE.match(s)):
                continue
            m = STEM_NUM_RE.match(s) or STEM_LOST_RE.match(s)
            numbered = bool(m)
            if m:
                s = s[m.end():].strip()
            if not s:
                continue
            if not items:
                stem = [s] if numbered else stem + [s]
            elif len(items) >= 4:
                had = bool(queue)
                flush()
                stem = stem + [s] if (had and not numbered) else [s]
            elif numbered:
                queue[:] = [s]                       # سؤال جاي قبل ما اختيارات اللي قبله تخلص
            else:
                items[-1] = (items[-1][0], (items[-1][1] + " " + s).strip())  # تكملة اختيار
    if items:
        flush()
    return out


# ---------- أوراق عربي: حروف أ ب ج د ----------
def _parse_arabic_labeled(text: str):
    """السؤال بيبدأ برقم (لو موجود) والاختيارات 4 سطور أولها أ ثم ب ج د.
    بيتحمل الأقواس الناقصة والأرقام العربي، ومش بيعتمد على الترقيم."""
    out, stem, raw_choices = [], [], []

    def emit():
        explicit = [bool(_ar_label_re(l).match(c)) for l, c in zip(AR_LABELS, raw_choices)]
        use_glued = sum(explicit) >= 2  # لو أغلب الأسطر عليها حروف، يبقى الباقي حرف ملزوق
        choices = []
        for l, c, e in zip(AR_LABELS, raw_choices, explicit):
            if e:
                c = _ar_label_re(l).sub("", c, count=1)
            elif use_glued and c.startswith(l):
                c = c[1:].lstrip(AR_PUNCT.replace("\\", ""))
            choices.append(c.strip())
        out.append((" ".join(stem).strip(), choices))
        stem.clear()
        raw_choices.clear()

    for raw in text.splitlines():
        for cell in raw.split("\t"):
            s = cell.strip()
            if not s or MARKS_RE.match(s):
                continue
            if raw_choices:  # جوه اختيارات سؤال
                raw_choices.append(s)
                if len(raw_choices) == 4:
                    emit()
                continue
            if stem and AR_A_RE.match(s):  # أول اختيار
                raw_choices.append(s)
                continue
            m = NUM_AR_RE.match(s)
            if m:  # رقم سؤال جديد
                stem.clear()
                s = s[m.end():].strip()
            if ARABIC_RE.search(s):
                stem.append(s)
    if raw_choices:  # اختيارات ناقصة: نسجلها وتتعلّم bad_choices
        emit()
    return out or None


# ---------- أوراق من غير ترقيم وحروف (قاعدة الأسطر القصيرة) ----------
def _is_short(text: str) -> bool:
    return len(text.split()) <= 4


def _parse_unnumbered(text: str):
    """السؤال سطور طويلة، واختياراته 4 خانات قصيرة (ممكن خانتين في السطر الواحد)."""
    out, stem, choices = [], [], []
    for line in text.splitlines():
        cells = [c.strip() for c in line.split("\t") if c.strip()]
        if not cells:
            continue
        has_long = any(not _is_short(x) for x in stem)

        if has_long and all(_is_short(c) for c in cells):
            for c in cells:
                if len(choices) == 4:  # اختيارات زيادة عن 4: الشكل مش واضح
                    return None
                choices.append(re.sub(r"^\(?[A-Da-dأبجد][\.\)\-]\s*", "", c))
            if len(choices) == 4:
                q = " ".join(stem)
                q = re.sub(r"^\s*[\d٠-٩]{1,3}\s*[\.\)\-]\s*", "", q).strip()
                out.append((q, choices))
                stem, choices = [], []
        else:
            if choices:  # اختيارات ناقصة وجه سطر طويل: الشكل مش واضح
                return None
            stem.append(" ".join(cells))
    if choices:
        return None
    return out or None


# ---------- الدوال الرئيسية ----------
def structure_page(raw_text: str, page: int, ocr_conf=None) -> list:
    if not raw_text.strip():
        return []

    # قسم المفتاح بيتفصل الأول عشان أرقامه ما تتحسبش أسئلة
    q_text, key_text = _split_key(raw_text)
    key = _parse_key(key_text) if key_text else {}
    has_key = bool(key)
    raw_text = q_text
    if not raw_text.strip():
        return []

    out = []

    ar_a_lines = sum(1 for c in re.split(r"[\n\t]", raw_text) if AR_A_RE.match(c.strip()))
    if ar_a_lines >= 2:
        parsed = _parse_arabic_labeled(raw_text)
        if parsed:
            for stem, choices in parsed:
                out.append(_build(stem, choices, "", page, ocr_conf, raw_text, has_key))
            return _apply_key(_flag_duplicates(out), key)

    if len(LABEL_RE.findall(raw_text)) >= 4:  # 2) ورقة بحروف A-D
        for stem, choices in _parse_stream(raw_text):
            out.append(_build(stem, choices, "", page, ocr_conf, raw_text, has_key))
    if not out:  # 3) ورقة من غير ترقيم
        cleaned = _clean_ocr(raw_text)
        parsed = _parse_unnumbered(cleaned)
        if parsed:
            for stem, choices in parsed:
                out.append(_build(stem, choices, "", page, ocr_conf, raw_text, has_key))
        elif re.search(r"[?؟]", cleaned):  # فيه علامات أسئلة بس الشكل مش مفهوم: للمراجعة
            out.append(_failed(cleaned, page, ocr_conf))
        # غير كده: ورقة من غير أسئلة اختيار من متعدد (تحليل، مقال...) فبنتجاهلها

    return _apply_key(_flag_duplicates(out), key)


def _find_page(q: dict, pages: list, start: int = 0) -> int:
    """رقم الصفحة اللي فيها بداية السؤال. بيدوّر من صفحة السؤال اللي قبله (الأسئلة بالترتيب)
    عشان نموذجين بنفس الأسئلة ما يتلخبطوش."""
    keys = [_norm(q["question"])[:15], _norm(q["choices"][0]) if q["choices"] else ""]
    for key in keys:
        if len(key) >= 3:
            for p in pages[start:] + pages[:start]:
                if key in _norm(p["raw_text"]):
                    return p["page"]
    return pages[start]["page"]


def structure_document(pages: list) -> list:
    """بيجمع نص كل الصفحات الأول، عشان السؤال اللي بيتقسم بين صفحتين يتفهم صح."""
    pages = [p for p in pages if p["raw_text"].strip()]
    if not pages:
        return []
    joined = "\n".join(p["raw_text"] for p in pages)
    conf = {p["page"]: p["ocr_confidence"] for p in pages}

    qs = structure_page(joined, pages[0]["page"], None)
    idx = {p["page"]: i for i, p in enumerate(pages)}
    start = 0
    for q in qs:
        q["page"] = _find_page(q, pages, start)
        start = idx[q["page"]]
        c = conf.get(q["page"])
        q["confidence"] = c
        if c is not None and c < 0.80 and "low_ocr_confidence" not in q["issues"]:
            q["issues"].append("low_ocr_confidence")
            q["needs_review"] = True
    return qs