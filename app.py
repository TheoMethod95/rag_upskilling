from fastapi import FastAPI, Query
from pydantic import BaseModel
import boto3
import os
from rag_pipeline import build_rag_pipeline, rag_query

app = FastAPI(title="RAG API")

# Load the pipeline once on startup
state = None

@app.on_event("startup")
def startup_event():
    global state
    state = build_rag_pipeline()

class QueryRequest(BaseModel):
    question: str

@app.post("/query")
def query_rag(req: QueryRequest):
    result = rag_query(req.question, state)
    return {
        "question": req.question,
        "answer": result["answer"],
        "citations": result["citations"]
    }
        
@app.post("/update")
def update_rag_pipeline():
    global state
    state = build_rag_pipeline()  # reloads all files + updates FAISS
    return {"status": "RAG pipeline updated!"}

PROFILE_NAME = "genai-bedrock"
REGION_NAME = "us-east-1"
S3_BUCKET = "my-bedrock-docs-123"

session = boto3.Session(profile_name=PROFILE_NAME, region_name=REGION_NAME)
s3 = session.client("s3")

@app.post("/delete")
def delete_rag_file(file_name: str = Query(..., description="S3 key of the file to delete")):
    global state

    #1. delete from s3
    s3.delete_object(Bucket=S3_BUCKET, Key=file_name)

    #2. wipe persistent index
    from rag_pipeline import EMBEDDINGS_FILE, FAISS_FILE
    
    if os.path.exists(EMBEDDINGS_FILE):
        os.remove(EMBEDDINGS_FILE)
    if os.path.exists(FAISS_FILE):
        os.remove(FAISS_FILE)
    
    #. rebuild pipeline from s3
    state = build_rag_pipeline()

    
    return {"status": "File deleted and reindexed",
            "file_name": file_name
    }
