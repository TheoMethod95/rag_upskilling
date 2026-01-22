import streamlit as st
import requests
import boto3
from io import BytesIO
import os
from upload_and_update import upload_and_update

#API_URL = "http://100.48.51.196:8000/query"
API_URL = "http://localhost:8000/query"


st.set_page_config(page_title="RAG Demo", layout="wide")
st.title("RAG Question Answering about Bolivian History")

tab = st.sidebar.radio("Navigation", ["Ask Question", "Upload New Data"])

if tab == "Ask Question":
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
                st.write(data["answer"])
                if "citations" in data and data["citations"]:
                    st.subheader("Citations")
                    for c in data["citations"]:
                        st.markdown(
                            f"- **{c['document']}** (words {c['start_word']}–{c['end_word']})"
                        )
            else:
                st.error(f"Error from API: {response.status_code} - {response.text}")

elif tab == "Upload New Data":
    st.subheader("Upload new .txt files to S3")
    uploaded_files = st.file_uploader("Drag and drop .txt files", accept_multiple_files=True, type=["txt"])

    if st.button("Upload and Update"):
        if not uploaded_files:
            st.warning("Please select files to upload.")
        else:
            # Save temp locally
            temp_paths = []
            for file in uploaded_files:
                temp_path = f"temp_uploads/{file.name}"
                os.makedirs("temp_uploads", exist_ok=True)
                with open(temp_path, "wb") as f:
                    f.write(file.getbuffer())
                temp_paths.append(temp_path)

            with st.spinner("Uploading files and updating RAG index..."):
                upload_and_update(temp_paths)
            st.success("Files uploaded and RAG index updated!")