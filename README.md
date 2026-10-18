# University Chatbot 🎓

A chatbot that answers questions about the **University of Peshawar** using the university's own PDF documents (admissions, fees, dates, PhD documents, forms and guides).

It shows the **source file and page number** with every answer, and when it cannot find something it says so instead of guessing. It runs offline and needs **no API key**.

## Features

- Reads PDF and TXT files from the `docs/` folder
- Chat screen built with Streamlit
- Shows the source file and page for every answer
- Says "could not find" when the topic is not in the documents, and shows the closest passages
- Understands the brochure layout: one answer per program (eligibility and seats) and one answer per fee-table row
- "Reload documents" button to add new files without restarting

## Example questions

Tested with the **BS Brochure 2026**:

| Question | Result |
|---|---|
| What is the eligibility for Physics? | F.Sc (Pre-Medical/Pre-Engg.) with at least 45% marks, 130 seats |
| What is the fee for Physics? | Rs 54,500 per semester (subsidized fee), Rs 70,000 (non-subsidized) |
| What is the last date to apply? | 27 September 2026 |
| What is the eligibility for Computer Science? | Not found, because the BS brochure has no Computer Science program.

## How it works

1. **Read:** every PDF in `docs/` is read page by page with `pypdf`.
2. **Split:** the text is cut into small pieces. Programs ("ELIGIBILITY ... TOTAL SEATS") and fee-table rows get one piece each. Normal text is cut into about 80-word pieces.
3. **Search:** each piece is turned into TF-IDF vectors (word-based and letter-group-based) with scikit-learn. The question is compared with all pieces using cosine similarity.
4. **Check:** the main words of the question (for example "computer science") must appear in the answer. If not, the bot says "not found" instead of showing an unrelated passage.
5. **Show:** the best piece is shown with its source file and page.

## Project files

```
university-chatbot/
├── app.py            # Streamlit chat screen
├── chatbot.py        # reading, splitting and search code
├── find_text.py      # helper: find where a word appears in your documents
├── requirements.txt
├── README.md
└── docs/             # put your PDF / TXT files here (not included)
```

## Setup

1. Install Python 3.9 or newer.
2. Install the libraries:
   ```
   pip install -r requirements.txt
   ```
3. Create a folder called `docs` next to `app.py` and put your university PDFs inside.
   The documents are public and can be downloaded from [uop.edu.pk](http://www.uop.edu.pk) (for example the BS Brochure). They are not included in this repository.
4. Run the app:
   ```
   streamlit run app.py
   ```

If you add or change files later, click **Reload documents** in the sidebar.

To check where a word appears in your documents:

```
python find_text.py computer
```

## Limitations

- **Scanned PDFs** (photos of pages) have no text and cannot be read without OCR.
- **Search is based on words, not meaning.** Questions work best when they use the same words as the document.
- Long text is cut into small pieces, so an answer can start in the middle of a sentence. Use the page number to read the full text.
- A question can return a related passage that is not the right one. For example, a question about PhD eligibility can return text about curriculum changes if the document has no eligibility section. Always check the source page for important decisions such as admission and fees.
- The brochure has two fee tables (subsidized and non-subsidized). Both are shown, so read the label.

## Future work

- Meaning-based search with sentence embeddings
- An LLM to write the final answer from the retrieved passages (full RAG)
- OCR for scanned documents
- Urdu and Pashto questions

## Built with

Python, scikit-learn, pypdf, Streamlit

## Disclaimer

This is a student project. Answers come from the documents you add and may be incomplete or out of date. Always confirm important information with the university.
