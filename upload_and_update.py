# upload_and_update.py
import boto3
import os
import requests
from rag_pipeline import build_rag_pipeline

# -----------------------------
# Config
# -----------------------------
PROFILE_NAME = "genai-bedrock"
REGION_NAME = "us-east-1"
S3_BUCKET = "my-bedrock-docs-123"

#UPDATE_API_URL = "http://localhost:8000/update"
UPDATE_API_URL = "http://3.234.222.103:8000/update"

session = boto3.Session(profile_name=PROFILE_NAME, region_name=REGION_NAME)
s3 = session.client("s3", region_name=REGION_NAME)

def upload_file_to_s3(file_path: str, s3_bucket: str, s3_key: str):
    """Upload a local file to S3"""
    print(f"Uploading {file_path} to s3://{s3_bucket}/{s3_key}")
    s3.upload_file(file_path, s3_bucket, s3_key)
    print("Upload complete!")

def upload_and_update(file_paths: list):
    """
    Upload multiple files to S3 and update the RAG index.
    """
    # 1️⃣ Upload all files
    for path in file_paths:
        if not path.endswith(".txt"):
            print(f"Skipping {path}: Only .txt files are allowed")
            continue
        filename = os.path.basename(path)
        upload_file_to_s3(path, S3_BUCKET, filename)

    # 2️⃣ Update RAG pipeline (incremental embeddings + FAISS)
    print("Updating RAG pipeline with new files...")
    requests.post(UPDATE_API_URL) #updates the RAG pipeline
    print("RAG pipeline updated!")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Upload .txt files to S3 and update RAG pipeline")
    parser.add_argument("files", nargs="+", help="List of local .txt files to upload")
    args = parser.parse_args()

    upload_and_update(args.files)
