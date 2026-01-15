import json
import boto3
import faiss
import numpy as np
import sys
import os

# -----------------------------
# Step 0: Config
# -----------------------------
PROFILE_NAME = "genai-bedrock"
REGION_NAME = "us-east-1"
S3_BUCKET = "my-bedrock-docs-123"
EMBEDDING_MODEL = "amazon.titan-embed-text-v2:0"
LLM_MODEL = "anthropic.claude-3-sonnet-20240229-v1:0"
# -----------------------------
FAISS_FILE = "data_store/faiss.index"
EMBEDDINGS_FILE = "data_store/embeddings.json"

os.makedirs("data_store", exist_ok=True)


def build_rag_pipeline():

    # -----------------------------
    # Step 1: Load files from S3 dynamically
    # -----------------------------
    session = boto3.Session(profile_name=PROFILE_NAME, region_name=REGION_NAME)
    s3 = session.client("s3")
    bedrock_runtime = session.client("bedrock-runtime")

    documents = []

    # List all objects in the bucket
    response = s3.list_objects_v2(Bucket=S3_BUCKET)
    all_objects = response.get("Contents", [])

    # Filter for .txt files
    txt_files = [obj for obj in all_objects if obj["Key"].endswith(".txt")]

    if not txt_files:
        raise ValueError(f"No .txt files found in bucket {S3_BUCKET}")

    for obj in txt_files:
        key = obj["Key"]
        last_modified = obj["LastModified"].isoformat()  # for freshness checks
        size = obj["Size"]

        s3_obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
        text = s3_obj["Body"].read().decode("utf-8")

        documents.append({
            "key": key,
            "text": text,
            "s3_last_modified": last_modified,
            "s3_size": size,
            "source": f"s3://{S3_BUCKET}/{key}"
        })

    print(f"Loaded {len(documents)} text files from S3:")
    for doc in documents:
        print(f"- {doc['key']} (size: {doc['s3_size']} bytes).")

    # -----------------------------
    # Step 2: Chunk documents
    # -----------------------------
    def chunk_text_with_metadata(text, chunk_size=20):
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunks.append({
                "chunk_text": " ".join(words[i:i+chunk_size]),
                "start_word": i,
                "end_word": min(i + chunk_size, len(words))
            })
        return chunks

    all_chunks = []
    for doc in documents:
        chunks = chunk_text_with_metadata(doc["text"])
        for idx, c in enumerate(chunks):
            all_chunks.append({
                "document_id": doc["key"],
                "chunk_id": idx,
                "chunk": c["chunk_text"],
                "start_word": c["start_word"],
                "end_word": c["end_word"],
                "source": f"s3://{S3_BUCKET}/{doc['key']}"
            })

    print("Total chunks:", len(all_chunks))

    # -----------------------------
    # Step 3: Generate embeddings (Titan v2) 
    # -----------------------------
    if os.path.exists(EMBEDDINGS_FILE):
        print("Found persisted embeddings")

        with open(EMBEDDINGS_FILE, "r") as f:
            embeddings = json.load(f)

    else:
        print("Persisted embeddings not found. Generating now.", flush=True)

        embeddings = []
        for item in all_chunks:
            payload = {
                "inputText": item["chunk"],
                "embeddingTypes": ["binary"]
            }
            
            response = bedrock_runtime.invoke_model(
                modelId=EMBEDDING_MODEL,
                body=json.dumps(payload),
                accept="application/json",
                contentType="application/json"
            )
            
            result = json.loads(response['body'].read())
            
            embeddings.append({
                "document_id": item["document_id"],
                "chunk_id": item["chunk_id"],
                "chunk": item["chunk"],
                "start_word": item["start_word"],
                "end_word": item["end_word"],
                "source": item["source"],
                "embedding": result["embeddingsByType"]["binary"]
            })


        with open(EMBEDDINGS_FILE, "w") as f:
            json.dump(embeddings, f)
        print("Saved embeddings for future runs", flush=True)

    # -----------------------------
    # Step 4: Create FAISS index
    # -----------------------------
    if os.path.exists(FAISS_FILE):
        print("Found persisted FAISS index", flush=True)
        index = faiss.read_index(FAISS_FILE)

    else:
        print("Persisted FAISS index not found. Generating now.", flush=True)

        dim = len(embeddings[0]["embedding"])
        index = faiss.IndexFlatL2(dim)
        vectors = np.array([e["embedding"] for e in embeddings]).astype("float32")
        index.add(vectors)
        print("FAISS index created with", index.ntotal, "vectors", flush=True)

        faiss.write_index(index, FAISS_FILE)

        print("FAISS index saved to disk.", flush=True)

    return {
        "index": index,
        "embeddings": embeddings,
        "bedrock_runtime": bedrock_runtime
    }

# -----------------------------
# Step 5: Query function
# -----------------------------
def rag_query(user_question, state, top_k=3):
    index = state["index"]
    embeddings = state["embeddings"]
    bedrock_runtime = state["bedrock_runtime"]
    
    # 1️⃣ Generate embedding for question
    payload = {
        "inputText": user_question,
        "embeddingTypes": ["binary"]
    }
    response = bedrock_runtime.invoke_model(
        modelId=EMBEDDING_MODEL,
        body=json.dumps(payload),
        accept="application/json",
        contentType="application/json"
    )
    result = json.loads(response['body'].read())
    query_vector = np.array(result['embeddingsByType']['binary']).astype("float32").reshape(1, -1)
    
    # 2️⃣ Retrieve top-k chunks from FAISS
    D, I = index.search(query_vector, top_k)

    retrieved_chunks = []
    citations = []

    for idx in I[0]:
        metadata = embeddings[idx]   # ✅ THIS is where it goes

        retrieved_chunks.append(
            f"[{metadata['document_id']}]\n{metadata['chunk']}"
        )

        citations.append({
            "document": metadata["document_id"],
            "source": metadata["source"],
            "chunk_id": metadata["chunk_id"],
            "start_word": metadata["start_word"],
            "end_word": metadata["end_word"]
        })
    
    # 3️⃣ Build prompt for Claude
    prompt = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Answer the question using ONLY the context below.\n"
                    "Return your response in the following format:\n\n"
                    "Answer:\n<one concise paragraph>\n\n"
                    "Sources:\n"
                    "- <document_id> (word range start_word–end_word)\n\n"
                    "If the answer cannot be found, say: 'Not found in provided documents.'\n\n"
                    "Context:\n"
                    f"{chr(10).join(retrieved_chunks)}\n\n"
                    f"Question: {user_question}"
                )
            }
        ],
        "max_tokens": 300
    }
    
    # 4️⃣ Invoke Claude
    response = bedrock_runtime.invoke_model(
        modelId=LLM_MODEL,
        body=json.dumps(prompt),
        accept="application/json",
        contentType="application/json"
    )
    answer = json.loads(response['body'].read())
    
    return {
    "answer": answer['content'][0]['text'],
    "citations": citations
}


