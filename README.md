# Question Bank – AI Pipeline

Converts exam **images or PDFs** (Arabic or English, multiple-choice) into **structured JSON** for the Backend:

```
file (image / PDF)  ->  text (OCR or direct PDF text)  ->  questions + 4 choices (+ answer from the answer key)  ->  JSON
```

The pipeline never touches the database. It returns (and saves) a JSON object; the Backend decides what to store.
`department`, `level` and `subject` are **not** part of this JSON – the Backend adds them when saving.

---

## 1. Requirements

- Windows, Python 3.10 (via Miniforge/conda)
- [Ollama](https://ollama.com) (local LLM, free, no API key) – used only as a fallback for layouts the rules cannot parse
- Optional: NVIDIA GPU. The OCR currently runs on CPU, which is enough for testing.

## 2. Installation

Run everything in **Miniforge Prompt**.

```bash
conda env create -f environment.yml
conda activate qbank-ai
```

If `environment.yml` fails, create the environment by hand:

```bash
conda create -n qbank-ai python=3.10 -y
conda activate qbank-ai
conda install -c conda-forge pymupdf pillow numpy python-dotenv pytest -y
python -m pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
python -m pip install paddleocr ollama
```

Then install Ollama from ollama.com and download the model once:

```bash
ollama pull qwen2.5:3b
```

Ollama must be running (tray icon) whenever the pipeline runs.

**First run:** PaddleOCR downloads its models (about 100–200 MB per language) and caches them in
`C:\Users\<you>\.paddlex`. If the connectivity check hangs, run this first (PowerShell):

```powershell
$env:PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK="True"
```

## 3. Usage

Run from the `ai_pipeline` folder.

### Command line

```bash
python -m src.pipeline path/to/file.pdf en        # English file
python -m src.pipeline path/to/file.jpg ar        # Arabic file
python -m src.pipeline path/to/folder en          # every image/PDF in a folder
```

The language is `en`, `ar` or `auto` (`auto` runs both OCR models and is slower – prefer `en`/`ar`).
Output is written to `output/<file name>.json`.

To re-run only the question-extraction step on an existing JSON (no OCR):

```bash
python -m src.restructure "output/file.json"
```

### From Python (Backend)

```python
from src.pipeline import run_pipeline

result = run_pipeline("uploads/midterm.pdf", lang="ar")   # returns a dict
```

`run_pipeline(file_path, lang="auto", output_path=None, use_gpu=True)` never raises on bad input; problems are
reported in `errors`. It also saves the JSON to `output/` (or `output_path`). Supported files: `.jpg .jpeg .png .bmp .webp .pdf`.

## 4. Output format

```json
{
  "status": "success",
  "source_file": "midterm.pdf",
  "pages": [
    { "page": 1, "raw_text": "1. Which of the following ...", "ocr_confidence": 0.97 }
  ],
  "questions": [
    {
      "question": "Which of the following is a supervised learning algorithm?",
      "choices": ["K-Means", "Linear Regression", "DBSCAN", "Apriori"],
      "correct_answer": "Linear Regression",
      "page": 1,
      "confidence": 0.97,
      "needs_review": false,
      "issues": []
    }
  ],
  "errors": []
}
```

| Field | Meaning |
|---|---|
| `status` | `success`, `partial` (some step failed) or `failed` (nothing usable; see `errors`) |
| `pages[].raw_text` | text read from each page |
| `pages[].ocr_confidence` | OCR confidence 0–1, or `null` when the PDF page had real text |
| `questions[].choices` | exactly 4 strings (empty only when `issues` contains `parse_failed`) |
| `questions[].correct_answer` | one of the choices, or `""` when unknown |
| `questions[].needs_review` | `true` when `issues` is not empty |
| `errors[]` | `{file, stage, message}` objects |

**Rule for the Backend: do not publish a question with `needs_review: true` before an admin reviews it.**

### `issues` values

| Value | Meaning |
|---|---|
| `no_answer` | no answer key in the file, or this question's answer was not readable |
| `low_ocr_confidence` | OCR confidence below 0.80 |
| `choices_not_in_text` | a choice does not appear in the page text (possible OCR error or invented text) |
| `bad_choices` | not exactly 4 different choices |
| `empty_question` / `short_question` | question text missing or too short |
| `duplicate_question` | same question text appears twice in the file |
| `answer_key_mismatch` | the answer key has a different number of answers than questions found, so no answer was applied |
| `parse_failed` | the block could not be understood; kept for the admin instead of being dropped |

## 5. How it works

1. **Read the file.** Image: OCR. PDF: pages with real text are read directly; scanned pages and any page containing
   Arabic are rendered to an image and OCR'd (direct Arabic text extraction loses letters such as the *lam* in *lam-alef*).
2. **OCR** (PaddleOCR) groups words into lines, in reading order (right-to-left for Arabic).
3. **Structure** all pages together (a question may continue on the next page):
   - Arabic papers with choices `أ ب ج د` – rules
   - numbered papers with choices `A B C D` – rules
   - papers without numbering – short-line rules
   - anything else – local LLM (Ollama), with every choice checked against the page text
4. **Answer key.** A section titled `Answer Key`, `Answers:`, `الإجابات`, `مفتاح`... is parsed (`1-B`, `1. B`, `1-ب`, `ب-1`)
   and applied by question order. Answers are **never guessed**; without a key `correct_answer` stays empty.

## 6. Project structure

```
ai_pipeline/
  src/
    ingestion/image_extractor.py   OCR for one image
    ingestion/pdf_extractor.py     PDF pages -> text or OCR
    extraction/structurer.py       text -> questions, answer key, issues
    pipeline.py                    run_pipeline() and the command line
    restructure.py                 re-run the question step on an existing JSON
  tests/samples/                   sample English files
  tests/samples_ar/                sample Arabic files
  output/                          generated JSON (not committed)
```

## 7. Known limits

- Answer keys in table form (numbers in one row, letters in the next) and tick marks beside choices are not read.
- Arabic OCR is weaker on low-quality photos: expect `low_ocr_confidence` and `choices_not_in_text`.
- Watermarks and underlines can add stray characters or remove blanks from questions; the admin review catches these.
- The local model (`qwen2.5:3b`) is small. Layouts handled by rules are reliable; the LLM fallback may miss or alter questions
  (this is why every LLM choice is checked against the page text). Change `MODEL` in `structurer.py` to try a larger model.
- OCR runs on CPU; GPU support is optional and not required.
