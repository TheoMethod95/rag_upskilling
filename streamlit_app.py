import streamlit as st
import requests
import re

API_URL = "http://100.48.51.196:8000/query"

st.set_page_config(page_title="RAG Demo", layout="wide")
st.title("RAG Question Answering about Bolivian History")

question = st.text_input("Ask a question about Bolivian History:")

def parse_answer(text):
    """
    Parse structured answer like:
    Source: bolivia_history.txt
    Response: Bolivia is...
    """
    source_match = re.search(r"Source:\s*(.+)", text, re.IGNORECASE)
    response_match = re.search(r"Response:\s*(.+)", text, re.IGNORECASE | re.DOTALL)

    source = source_match.group(1).strip() if source_match else "No source provided"
    response = response_match.group(1).strip() if response_match else text.strip()
    
    return source, response

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    API_URL,
                    json={"question": question},
                    timeout=120
                )
            except requests.exceptions.RequestException as e:
                st.error(f"Error calling API: {e}")
                st.stop()

        if response.status_code == 200:
            data = response.json()
            answer_text = data.get("answer", "")

            # Parse structured response
            source, answer = parse_answer(answer_text)

            st.subheader("Source")
            st.write(source)

            st.subheader("Answer")
            st.write(answer)

            if "citations" in data:
                st.subheader("Citations")
                for c in data["citations"]:
                    st.write(
                        f"- **{c['document']}** "
                        f"(words {c['start_word']}–{c['end_word']})"
                    )
        else:
            st.error(f"Error from API: {response.status_code} - {response.text}")
