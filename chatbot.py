"""
University chatbot - search engine part (no AI model, no API key, works offline).

How it works:
1. Read every PDF / TXT file in the docs/ folder, page by page.
2. Cut the text into small pieces:
   - one piece per program ("ELIGIBILITY ... TOTAL SEATS : 130")
   - one piece per fee-table row
   - ~80-word pieces for normal text
3. Score every piece against the question (word TF-IDF + letter-group TF-IDF).
4. Return the best piece. If the main words of the question (for example
   "computer science") are not in any piece, say "not found" instead of guessing.
"""
import os
import re
from dataclasses import dataclass

from pypdf import PdfReader
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CHUNK_WORDS = 80        # size of normal text pieces
OVERLAP_WORDS = 25      # pieces overlap so answers are not cut in half
TOP_K = 3               # how many passages to return
MIN_SCORE = 0.10        # below this, we say "not found"
MIN_CHUNK_WORDS = 20    # ignore very short pieces (headings, cover and divider pages)
MAX_SHOWN_CHARS = 700   # longest answer text shown
MIN_COVERAGE = 0.6      # share of the question's main words that must appear

# words that appear in most questions and do not identify the topic
GENERIC_WORDS = {
    "eligibility", "eligible", "bs", "program", "programs", "programme", "degree",
    "admission", "admissions", "seats", "seat", "total", "fee", "fees", "structure",
    "marks", "required", "requirement", "requirements", "university", "tell", "need",
    "many", "much", "cost", "costs", "number", "does", "peshawar", "uop",
}


@dataclass
class Chunk:
    text: str
    source: str   # file name
    page: int     # page number (1 for txt files)


# ---------------------------------------------------------------- reading ---
def _split_words(text, size=CHUNK_WORDS, overlap=OVERLAP_WORDS):
    words = text.split()
    if not words:
        return []
    step = max(size - overlap, 1)
    return [" ".join(words[i:i + size]) for i in range(0, len(words), step)
            if words[i:i + size]]


_SEATS = re.compile(r"(TOTAL\s+SEATS\s*:\s*\d+)", re.IGNORECASE)

# fee-table rows:  "29. Physics 3000 45100 4300 1600 500 54500 Semester"
_ROW = re.compile(
    r"(\d{1,2})\.\s+([A-Za-z][A-Za-z &/().\-]*?)\s+((?:\d{3,6}\s+){5}\d{3,6})(\s+Semester)?")


def _fee_pieces(text, state):
    """Turn fee-table rows into readable sentences. Returns (pieces, leftover_text)."""
    if text.count("Semester") < 5 and "Fee for 2nd" not in text:
        return [], text
    pieces, leftover, last = [], [], 0
    for m in _ROW.finditer(text):
        number = int(m.group(1))
        name = re.sub(r"\s+", " ", m.group(2)).strip(" -")
        n = [int(x) for x in m.group(3).split()]
        if m.group(4):   # normal rows end with the word "Semester"
            if number == 1:
                state["tables"] = state.get("tables", 0) + 1
            kind = "subsidized fee" if state.get("tables", 1) <= 1 else "non-subsidized fee"
            pieces.append(
                f"Fee for {name} ({kind}): total Rs {n[5]} per semester "
                f"(admission and semester renewal fee {n[0]}, tuition fee {n[1]}, "
                f"misc fee {n[2]}, S&S {n[3]}, CMS {n[4]}).")
        else:            # "new programs" table has a different set of columns
            pieces.append(
                f"Fee for {name} (new program, regular fee): tuition fee Rs {n[0]}, "
                f"misc fee {n[1]}, S&S {n[2]}, CMS {n[3]}, security {n[4]} (refundable), "
                f"fee for 2nd semester onwards Rs {n[5]}.")
        leftover.append(text[last:m.start()])
        last = m.end()
    leftover.append(text[last:])
    return pieces, " ".join(leftover)


def _pieces(text, state):
    """Split one page into pieces (programs, fee rows, or word chunks)."""
    fee_rows, text = _fee_pieces(text, state)
    parts = _SEATS.split(text)           # [text, 'TOTAL SEATS : 130', text, ...]
    if len(parts) >= 5:                  # at least two programs on the page
        entries = []
        for i in range(0, len(parts) - 1, 2):
            entry = (parts[i] + " " + parts[i + 1]).strip()
            if entry:
                entries.append(entry)
        rest = parts[-1].strip()
        if rest:
            entries.append(rest)
        return fee_rows + entries
    return fee_rows + _split_words(text)


# dot leaders / table-of-contents lines
_LEADER = re.compile(r"(?:[.\u2024\u2025\u2026\u00b7\u22ef\u30fb]\s*){5,}")


def _is_junk(piece):
    if _LEADER.search(piece):
        return True
    no_space = piece.replace(" ", "")
    letters_digits = sum(ch.isalnum() for ch in no_space)
    return len(no_space) > 0 and letters_digits / len(no_space) < 0.75


def load_documents(folder="docs"):
    """Return (chunks, skipped_files). skipped_files = files with no readable text."""
    os.makedirs(folder, exist_ok=True)
    chunks, skipped = [], []
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        ext = name.lower().rsplit(".", 1)[-1] if "." in name else ""
        pages = []
        try:
            if ext == "pdf":
                reader = PdfReader(path)
                pages = [(i + 1, p.extract_text() or "") for i, p in enumerate(reader.pages)]
            elif ext == "txt":
                with open(path, encoding="utf-8", errors="ignore") as f:
                    pages = [(1, f.read())]
            else:
                continue
        except Exception:
            skipped.append(name)
            continue

        found, state = False, {}
        for page_no, text in pages:
            text = re.sub(r"\s+", " ", text).strip()
            text = re.sub(r"^\d{1,2}\s+(?=\d{1,2}\.\s)", "", text)   # leading page number
            for piece in _pieces(text, state):
                is_special = piece.startswith("Fee for") or _SEATS.search(piece)
                if len(piece.split()) < (6 if is_special else MIN_CHUNK_WORDS):
                    continue                    # headings / divider pages
                if _is_junk(piece):
                    continue                    # table of contents
                chunks.append(Chunk(piece, name, page_no))
                found = True
        if not found:
            skipped.append(name)   # probably a scanned PDF (images only)
    return chunks, skipped


# ----------------------------------------------------------------- search ---
def _main_words(question):
    words = re.findall(r"[a-z]+", question.lower())
    return [w for w in words
            if len(w) > 2 and w not in ENGLISH_STOP_WORDS and w not in GENERIC_WORDS]


def _covers(word, text_lower):
    return word[:4] in text_lower      # "apply" matches "application", "science" matches "sciences"


def _coverage(words, text):
    if not words:
        return 1.0
    t = text.lower()
    return sum(_covers(w, t) for w in words) / len(words)


class SearchIndex:
    def __init__(self, chunks):
        self.chunks = chunks
        if chunks:
            texts = [c.text for c in chunks]
            self.word_vec = TfidfVectorizer(
                stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
            self.char_vec = TfidfVectorizer(
                analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
            self.word_m = self.word_vec.fit_transform(texts)
            self.char_m = self.char_vec.fit_transform(texts)

    def search(self, question, k=30):
        """Return list of (score, Chunk) sorted best first."""
        if not self.chunks:
            return []
        s = 0.5 * cosine_similarity(self.word_vec.transform([question]), self.word_m)[0] \
            + 0.5 * cosine_similarity(self.char_vec.transform([question]), self.char_m)[0]
        q = question.lower()
        fee_intent = any(w in q for w in ("fee", "cost", "tuition", "pay", "charges"))
        elig_intent = any(w in q for w in ("eligib", "seats", "requirement", "qualif"))
        for i, c in enumerate(self.chunks):
            if fee_intent and c.text.startswith("Fee for"):
                s[i] += 0.05
            if elig_intent and _SEATS.search(c.text):
                s[i] += 0.05
        best = s.argsort()[::-1][:k]
        return [(float(s[i]), self.chunks[i]) for i in best]


def best_sentences(question, passage, n=2):
    """Show the passage itself (trimmed if very long). Cutting a table or a list
    into 'best sentences' made answers confusing."""
    if len(passage) <= MAX_SHOWN_CHARS:
        return passage
    return passage[:MAX_SHOWN_CHARS].rsplit(" ", 1)[0] + "..."


def answer(question, index):
    """Return dict: found, answer, source, page, others, missing."""
    results = index.search(question)
    words = _main_words(question)
    covering = [(s, c) for s, c in results if _coverage(words, c.text) >= MIN_COVERAGE]
    if not covering or covering[0][0] < MIN_SCORE:
        return {"found": False, "answer": "", "source": "", "page": 0,
                "others": results[:TOP_K], "missing": words}
    score, top = covering[0]
    return {
        "found": True,
        "answer": best_sentences(question, top.text),
        "source": top.source,
        "page": top.page,
        "others": covering[1:TOP_K],
        "missing": [],
    }
