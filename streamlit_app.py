import streamlit as st
import requests

API_URL = "http://100.48.51.196:8000/query"

st.set_page_config(page_title="RAG Demo", layout="wide")
st.title("RAG Question Answering about Bolivian History")

question = st.text_input("Ask a question about Bolivian History:")

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            response = requests.post(
                API_URL,
                json={"question": question},
                timeout=120
            )

        if response.status_code == 200:
            data = response.json()
            
            st.subheader("Answer")
            # if answer is a dict
            if isinstance(data["answer"], dict):
                st.write(data["answer"]["answer"])
            else:
                st.write(data["answer"])

            if "citations" in data and data["citations"]:
                st.subheader("Citations")
                for c in data["citations"]:
                    st.write(
                        f"- **{c['document']}** "
                        f"(words {c['start_word']}–{c['end_word']})"
                    )
        else:
            st.error(f"Error from API: {response.status_code} - {response.text}")
