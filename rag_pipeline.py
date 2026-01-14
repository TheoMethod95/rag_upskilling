import json
import boto3
import faiss
import numpy as np

# -----------------------------
# Step 0: Config
# -----------------------------
PROFILE_NAME = "genai-bedrock"
REGION_NAME = "us-east-1"
S3_BUCKET = "my-bedrock-docs-123"  # replace with your bucket
FILE_KEYS = ["bolivia_history.txt", "bolivia_geography.txt"]
EMBEDDING_MODEL = "amazon.titan-embed-text-v2:0"
LLM_MODEL = "anthropic.claude-3-sonnet-20240229-v1:0"

# -----------------------------
# Step 1: Load files from S3
# -----------------------------
session = boto3.Session(profile_name=PROFILE_NAME, region_name=REGION_NAME)
s3 = session.client("s3")
bedrock_runtime = session.client("bedrock-runtime")

documents = []
for key in FILE_KEYS:
    obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    text = obj["Body"].read().decode("utf-8")
    documents.append({"key": key, "text": text})

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


print("Generated embeddings for all chunks.")

# -----------------------------
# Step 4: Create FAISS index
# -----------------------------
dim = len(embeddings[0]["embedding"])
index = faiss.IndexFlatL2(dim)
vectors = np.array([e["embedding"] for e in embeddings]).astype("float32")
index.add(vectors)
print("FAISS index created with", index.ntotal, "vectors")

# -----------------------------
# Step 5: Query function
# -----------------------------
def rag_query(user_question, top_k=3):
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
                    "If you use information from the context, cite the document name.\n\n"
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
    return answer['content'][0]['text']

# -----------------------------
# Step 6: Test RAG query
# -----------------------------
question = "What is the capital of Bolivia and some geographic features?"
answer = rag_query(question)
print("Answer:\n", answer)
