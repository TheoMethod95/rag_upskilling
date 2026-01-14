# AWS Bedrock RAG with Citations

This project demonstrates a production-style Retrieval-Augmented Generation (RAG)
pipeline using AWS Bedrock, Titan Embeddings v2, FAISS, and Claude 3.



## Features

- Automatically loads all `.txt` files from an S3 bucket.
- Chunks documents and generates embeddings using **Titan v2**.
- Stores embeddings and FAISS index locally for fast retrieval.
- Serves a REST API using **FastAPI** with Swagger docs.
- Deployable locally or on **AWS EC2**.

## Prerequisites

- Python 3.12.7  
- AWS account with **Bedrock** and **S3** access  
- S3 bucket containing `.txt` documents  
- EC2 key pair (if deploying on EC2)  

## Setup

1. Clone the repository:
```bash
    git clone https://github.com/your-username/rag-upskilling.git
    cd rag-upskilling
```

2. Create a virtual environment (optional but recommended):
```bash
    python3 -m venv venv
source venv/bin/activate
```

3.Upgrade pip and install dependencies:
```bash
    pip install --upgrade pip
    pip install -r requirements.txt
```

4. Configure AWS credentials:
```bash
    aws configure
```

5. Set environment variables on EC2:
```bash
    mkdir -p ~/.aws
    nano ~/.aws/credentials

    Add the following:
    [default]
    aws_access_key_id = your_access_key_id
    aws_secret_access_key = your_secret_access_key

    Optionally, create ~/.aws/config:
    [default]
region = us-east-1
output = json
```


## Usage

- Query the RAG pipeline via HTTP:
```bash
    curl http://localhost:8000/query -X POST -H "Content-Type: application/json" -d '{"question": "What is the capital of Bolivia?"}'
```

- Access Swagger docs:
    http://localhost:8000/docs  