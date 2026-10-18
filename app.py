"""
University Chatbot - Streamlit app
Run:  streamlit run app.py
Put your university PDF/TXT files inside the docs/ folder first.
"""
import streamlit as st
from chatbot import SearchIndex, answer, load_documents

st.set_page_config(page_title="University Chatbot", page_icon="🎓")
st.title("🎓 University Chatbot")
st.caption("Ask about admissions, fees, rules, timetables and more. "
           "Answers come only from the university documents.")


@st.cache_resource
def get_index():
    chunks, skipped = load_documents("docs")
    return SearchIndex(chunks), chunks, skipped


index, chunks, skipped = get_index()

with st.sidebar:
    st.header("Documents")
    files = sorted({c.source for c in chunks})
    if files:
        for f in files:
            st.write("📄 " + f)
    else:
        st.warning("No documents found. Put PDF or TXT files in the docs folder, "
                   "then click Reload.")
    if skipped:
        st.error("Could not read text from: " + ", ".join(skipped) +
                 " (scanned PDFs need OCR).")
    if st.button("Reload documents"):
        st.cache_resource.clear()
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Hello! Ask me anything about the university. "
                   "For example: What is the fee structure? What is the attendance policy?"}]

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

question = st.chat_input("Type your question...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    result = answer(question, index)
    if result["found"]:
        reply = f"{result['answer']}\n\n*Source: {result['source']}, page {result['page']}*"
    else:
        what = " ".join(result["missing"]) if result["missing"] else "this"
        reply = (f"Sorry, I could not find **{what}** in the university documents. "
                 "It may not be in the files you added, or it may use different words. "
                 "Try other words or contact the relevant office.")

    with st.chat_message("assistant"):
        st.markdown(reply)
        if result["others"]:
            title = "More related passages" if result["found"] else "Closest passages (probably not what you asked)"
            with st.expander(title):
                for score, c in result["others"]:
                    st.markdown(f"**{c.source}, page {c.page}**\n\n{c.text}")
    st.session_state.messages.append({"role": "assistant", "content": reply})
