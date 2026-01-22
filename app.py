# app.py
from fastapi import FastAPI
from pydantic import BaseModel
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
