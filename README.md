# AWS Bedrock RAG with Citations

This project demonstrates a production-style Retrieval-Augmented Generation (RAG)
pipeline using AWS Bedrock, Titan Embeddings v2, FAISS, and Claude 3.

## Architecture
- Documents stored in Amazon S3
- Text chunking with metadata preservation
- Vector embeddings via Amazon Titan Text Embeddings v2
- FAISS used for similarity search
- Claude 3 Sonnet generates answers with document-level citations
- FastAPI exposes the RAG pipeline as an HTTP service

## Features
- End-to-end RAG pipeline
- Document and chunk-level citations

