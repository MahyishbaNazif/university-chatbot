"""
Debug helper: show where a word appears in your documents.
Usage:  python find_text.py computer
"""
import sys
from chatbot import load_documents

word = " ".join(sys.argv[1:]).lower()
if not word:
    print("Usage: python find_text.py <word>")
    raise SystemExit

chunks, skipped = load_documents("docs")
print(len(chunks), "pieces loaded. Unreadable files:", skipped)
hits = [c for c in chunks if word in c.text.lower()]
print(len(hits), "pieces contain", repr(word))
for c in hits[:10]:
    i = c.text.lower().find(word)
    print("-", c.source, "page", c.page, "|", c.text[max(0, i - 80): i + 160])
