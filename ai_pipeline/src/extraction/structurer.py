import json
import re
import ollama

MODEL = "qwen2.5:3b"   # لو احتجت الـ LLM في العربي جرّب "qwen2.5:7b"

PROMPT = """You are given raw OCR text of a multiple-choice exam.
Extract every multiple-choice question.

Rules:
- Keep the original language of the text (Arabic or English). Do NOT translate.
- Fix obvious OCR mistakes (broken letters, stray symbols) but NEVER invent text.
- Each question must have exactly 4 choices, without the letters (A, B, C, D / a, b, c, d / أ، ب، ج، د).
- Copy the choices exactly as they appear in the text. Never add or change a choice, and never reuse choices from another question.
- "correct_answer" must be copied exactly from one of the choices, ONLY if the answer is clearly printed in the text. Otherwise use "" (empty string). Never guess.
- Only extract real exam questions that are followed by answer choices. Lab results, tables, forms, headers, and normal paragraphs are NOT questions.
- Questions may NOT be numbered. A question is a stem (possibly preceded by a quoted line of poetry or a passage) followed by exactly 4 short choice lines. Ignore marks lines such as "درجتان" and any garbage lines.
- If there are no questions, return {"questions": []}.
- Return JSON only, in this shape:
{"questions": [{"question": "...", "choices": ["...", "...", "...", "..."], "correct_answer": "..."}]}

OCR TEXT:
"""

ARABIC_RE = re.compile(r"[ء-ي]")
NUM_START_RE = re.compile(r"\s*\d{1,3}\s*[\.\)]\s")
LABEL_RE = re.compile(r"(?<!\w)\(?([A-Da-d])[\.\)]\s*")
MARKS_RE = re.compile(
    r"^[\W_]*(?:\d+\s*)?(?:درجتان|درجتين|درجة|درجه|درجات|marks?)[\W_]*$", re.IGNORECASE)

AR_LABELS = "أبجد"
AR_PUNCT = "\\)\\.\\-\\s\"'“”"
NUM_AR_RE = re.compile(r"^\s*[\d٠-٩]{1,3}\s*[\.\)\-،:]?(?:\s+|$|(?=[ء-ي]))")

# سطر بيبدأ بعنوان مفتاح الإجابات (إنجليزي أو عربي)
KEY_MARK_RE = re.compile(
    r"(?im)^[^\w\n]*(?:answer\s*keys?|answers?|model\s*answers?|correct\s*answers?"
    r"|الإجابات|الاجابات|الإجابة\s*النموذجية|الاجابة\s*النموذجية|مفتاح)")
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


def _parse(content: str) -> list:
    content = re.sub(r"^```(?:json)?|```$", "", content.strip(), flags=re.MULTILINE).strip()
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if not m:
            raise
        data = json.loads(m.group(0))
    return data.get("questions", []) if isinstance(data, dict) else []


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

    # إجابة الـ LLM تُقبل بس لو الورقة فيها مفتاح إجابات
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
    """يفصل نص الأسئلة عن قسم المفتاح (لو فيه). يرجّع (نص الأسئلة، نص المفتاح أو '')."""
    m = KEY_MARK_RE.search(raw_text)
    if not m:
        return raw_text, ""
    return raw_text[:m.start()], raw_text[m.start():]


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


# ---------- أوراق مرقمة: تقسيم بالقواعد (حروف A-D) ----------
def _split_blocks(text: str) -> list:
    parts = re.split(r"(?m)^(?=\s*\d{1,3}\s*[\.\)]\s)", text)
    return [p for p in parts if NUM_START_RE.match(p)]


def _parse_labeled(block: str):
    """بلوك فيه اختيارات A-D. يرجّع [(stem, [4 choices]), ...] أو None لو الشكل مش واضح."""
    questions = []
    stem, pending, choices = [], [], {}
    seen = False

    def close() -> bool:
        if set(choices) != set("ABCD") or not all(choices.values()):
            return False
        s = re.sub(r"^\s*\d{1,3}\s*[\.\)]\s*", "", " ".join(stem)).strip()
        questions.append((s, [choices[l] for l in "ABCD"]))
        return True

    for line in block.splitlines():
        line = line.strip()
        if not line:
            continue
        ms = list(LABEL_RE.finditer(line))

        if not ms:  # سطر من غير حروف اختيارات
            if not seen:
                stem.append(line)
            elif len(choices) < 4:  # تكملة اختيار مكتوب على سطرين
                last = list(choices)[-1]
                choices[last] = (choices[last] + " " + line).strip()
            else:  # بعد آخر اختيار: header أو سؤال ضاع ترقيمه
                pending.append(line)
            continue

        head = line[:ms[0].start()].strip()
        if head:
            if not seen:
                stem.append(head)
            elif len(choices) < 4:
                last = list(choices)[-1]
                choices[last] = (choices[last] + " " + head).strip()
            else:
                pending.append(head)

        for i, m in enumerate(ms):
            letter = m.group(1).upper()
            end = ms[i + 1].start() if i + 1 < len(ms) else len(line)
            text = line[m.end():end].strip()
            if letter in choices:  # الحروف بدأت من الأول = سؤال جديد
                if len(choices) < 4 or not close():
                    return None
                stem, choices, pending = pending, {}, []
            choices[letter] = text
            seen = True

    if not close():
        return None
    return questions or None


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


# ---------- الـ LLM (احتياطي للحالات اللي القواعد ماعرفتهاش) ----------
def _ask(chunk: str) -> list:
    resp = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT + chunk}],
        format="json",
        options={"temperature": 0, "num_ctx": 4096, "num_predict": 1024},
    )
    try:
        return _parse(resp["message"]["content"])
    except Exception:
        print("Could not parse model output:", resp["message"]["content"][:200])
        return []


def _from_llm(chunk, page, ocr_conf, raw_text, has_key) -> list:
    out = []
    for q in _ask(chunk):
        if isinstance(q, dict) and len(q.get("choices", [])) >= 2:
            out.append(_build(q.get("question", ""), q.get("choices", []),
                              q.get("correct_answer", ""), page, ocr_conf, raw_text, has_key))
    return out


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

    # 1) ورقة عربي بحروف أ ب ج د
    ar_a_lines = sum(1 for c in re.split(r"[\n\t]", raw_text) if AR_A_RE.match(c.strip()))
    if ar_a_lines >= 2:
        parsed = _parse_arabic_labeled(raw_text)
        if parsed:
            for stem, choices in parsed:
                out.append(_build(stem, choices, "", page, ocr_conf, raw_text, has_key))
            return _apply_key(_flag_duplicates(out), key)

    blocks = _split_blocks(raw_text)

    if len(blocks) >= 2:  # 2) ورقة مرقمة بحروف A-D
        for b in blocks:
            parsed = _parse_labeled(b)
            if parsed:
                for stem, choices in parsed:
                    out.append(_build(stem, choices, "", page, ocr_conf, raw_text, has_key))
                continue
            got = _from_llm(_clean_ocr(b), page, ocr_conf, raw_text, has_key)
            out += got or [_failed(b, page, ocr_conf)]
    else:  # 3) ورقة من غير ترقيم
        cleaned = _clean_ocr(raw_text)
        parsed = _parse_unnumbered(cleaned)
        if parsed:
            for stem, choices in parsed:
                out.append(_build(stem, choices, "", page, ocr_conf, raw_text, has_key))
        else:
            got = _from_llm(cleaned, page, ocr_conf, raw_text, has_key)
            out += got or [_failed(cleaned or raw_text, page, ocr_conf)]

    return _apply_key(_flag_duplicates(out), key)


def _find_page(q: dict, pages: list) -> int:
    """أول صفحة فيها بداية السؤال، وإلا أول اختيار."""
    keys = [_norm(q["question"])[:15], _norm(q["choices"][0]) if q["choices"] else ""]
    for key in keys:
        if len(key) >= 3:
            for p in pages:
                if key in _norm(p["raw_text"]):
                    return p["page"]
    return pages[0]["page"]


def structure_document(pages: list) -> list:
    """بيجمع نص كل الصفحات الأول، عشان السؤال اللي بيتقسم بين صفحتين يتفهم صح."""
    pages = [p for p in pages if p["raw_text"].strip()]
    if not pages:
        return []
    joined = "\n".join(p["raw_text"] for p in pages)
    conf = {p["page"]: p["ocr_confidence"] for p in pages}

    qs = structure_page(joined, pages[0]["page"], None)
    for q in qs:
        q["page"] = _find_page(q, pages)
        c = conf.get(q["page"])
        q["confidence"] = c
        if c is not None and c < 0.80 and "low_ocr_confidence" not in q["issues"]:
            q["issues"].append("low_ocr_confidence")
            q["needs_review"] = True
    return qs