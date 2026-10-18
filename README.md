# University Chatbot

Ask questions about your university and get answers taken from the official
documents (PDF/TXT), with the file name and page number as the source.
No API key, no internet needed after install.

## Setup
1. `pip install -r requirements.txt`
2. Create a folder called `docs` next to `app.py` and put your university
   PDF or TXT files inside (admission rules, fee structure, handbook, timetable...).
3. `streamlit run app.py`

If you add or change files later, click **Reload documents** in the sidebar.

## How it works
1. Reads every file in `docs/` page by page.
2. Cuts the text into small chunks (about 120 words).
3. Turns chunks into TF-IDF vectors (a score for how important each word is).
4. For each question it finds the most similar chunk and shows its best sentences.

## Limits
- Scanned PDFs (photos of pages) have no text and need OCR first.
- Search is based on matching words, so asking with the same words as the
  document works best.

## Ideas to improve
- Use sentence embeddings (sentence-transformers) for meaning-based search.
- Let an LLM write the final answer from the retrieved passages (full RAG).
