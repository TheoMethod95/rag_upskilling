import json
import boto3
import faiss
import numpy as np
import sys
import os
from PyPDF2 import PdfReader
import io
import re
import yaml

with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

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
# -----------------------------
BANNED_PATTERNS = {
    k: [re.compile(p, re.IGNORECASE) for p in v]
    for k, v in config["banned_patterns"].items()
}
BLOCKED_RESPONSES = config["blocked_responses"]
CLASSIFIER_LABELS = list(config["classifier_labels"])

os.makedirs("data_store", exist_ok=True)


def hardcoded_guardrail_check(text: str):
    for category, patterns in BANNED_PATTERNS.items():
        for pattern in patterns:
            if pattern.search(text):
                return category
    return "benign"

def classify_intent(user_question, bedrock_runtime):
    labels_formatted = "\n".join(f"- {label}" for label in CLASSIFIER_LABELS)

    classifier_prompt = {
    "anthropic_version": "bedrock-2023-05-31",
    "messages": [
        {
            "role": "user",
            "content": (
                "Classify the intent of the following user question.\n\n"
                "Choose EXACTLY one and only one label from this list:\n"
                f"{labels_formatted}\n\n"
                "Respond with ONLY the label.\n\n"
                f"Question:\n{user_question}"
            )
        }
    ],
    "max_tokens": 20
    }

    response = bedrock_runtime.invoke_model(
        modelId=LLM_MODEL,
        body=json.dumps(classifier_prompt),
        accept="application/json",
        contentType="application/json"
    )

    result = json.loads(response["body"].read())
    label = result["content"][0]["text"].strip().lower()

    # Defensive parsing
    label = label.split()[0].replace(".", "").strip().lower()


    if label not in CLASSIFIER_LABELS:
        label = "benign"

    return label


def build_rag_pipeline():

    # -----------------------------
    # Step 1: Load files from S3 dynamically
    # -----------------------------
    session = boto3.Session(profile_name=PROFILE_NAME, region_name=REGION_NAME)
    s3 = session.client("s3")
    bedrock_runtime = session.client("bedrock-runtime")

    

    # List all objects in the bucket
    response = s3.list_objects_v2(Bucket=S3_BUCKET)
    all_objects = response.get("Contents", [])

    # Filter for compatible files
    supported_extensions = (".txt", ".pdf")
    files_to_process = [obj for obj in all_objects if obj["Key"].lower().endswith(supported_extensions)]

    if not files_to_process:
        raise ValueError(f"No compatible files (.txt/.pdf) found in bucket {S3_BUCKET}")

    documents = []
    for obj in files_to_process:
        key = obj["Key"]
        last_modified = obj["LastModified"].isoformat()  # for freshness checks
        size = obj["Size"]

        s3_obj = s3.get_object(Bucket=S3_BUCKET, Key=key)

        if key.lower().endswith(".txt"):
            text = s3_obj["Body"].read().decode("utf-8")
        elif key.lower().endswith(".pdf"):
            pdf_bytes = s3_obj["Body"].read()
            pdf_stream = io.BytesIO(pdf_bytes)
            reader = PdfReader(pdf_stream)
            text = "\n".join(
                page.extract_text() or "" for page in reader.pages
            )

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
    def chunk_text_with_metadata(text, chunk_size=200):
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
        embedded_chunks = {
            (e["document_id"], e["chunk_id"]) for e in embeddings
        }
        print(f"Found existing embeddings for {len(embedded_chunks)} chunks.")
    else:
        embeddings = []
        embedded_chunks = set()
        print("No existing embeddings found. Starting fresh.")
    
    # -----------------------------
    # Step 3b: Generate embeddings for NEW chunks only
    # -----------------------------
    new_embeddings = []

    for item in all_chunks:
        if (item["document_id"], item["chunk_id"]) in embedded_chunks:
            continue  # skip already embedded
        payload = {
            "inputText": item["chunk"],
            "embeddingTypes": ["float"]
        }

        response = bedrock_runtime.invoke_model(
            modelId=EMBEDDING_MODEL,
            body=json.dumps(payload),
            accept="application/json",
            contentType="application/json"
        )

        result = json.loads(response['body'].read())

        new_embeddings.append({
            "document_id": item["document_id"],
            "chunk_id": item["chunk_id"],
            "chunk": item["chunk"],
            "start_word": item["start_word"],
            "end_word": item["end_word"],
            "source": item["source"],
            "embedding": result["embeddingsByType"]["float"]
        })

    if new_embeddings:
        print(f"Generated {len(new_embeddings)} new embeddings")
        embeddings.extend(new_embeddings)

        # Save embeddings
        with open(EMBEDDINGS_FILE, "w") as f:
            json.dump(embeddings, f)
    else:
        print("No new embeddings to generate")

    # -----------------------------
    # Step 4: Update or create FAISS index
    # -----------------------------
    if not embeddings:
        raise RuntimeError("No embeddings available to build FAISS index.")

    dim = len(embeddings[0]["embedding"])

    # Convert embeddings to numpy
    def to_vectors(emb_list):
        return np.array([e["embedding"] for e in emb_list]).astype("float32")

    if os.path.exists(FAISS_FILE):
        index = faiss.read_index(FAISS_FILE)
        print("Loaded existing FAISS index")

        if new_embeddings:
            vectors = to_vectors(new_embeddings)
            faiss.normalize_L2(vectors)

            # IDs must match positions in embeddings list
            start_id = len(embeddings) - len(new_embeddings)
            ids = np.arange(start_id, start_id + len(new_embeddings))

            index.add_with_ids(vectors, ids)
            faiss.write_index(index, FAISS_FILE)
            print("Updated FAISS index with new embeddings")

    else:
        print("Creating new FAISS index")

        index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))

        vectors = to_vectors(embeddings)
        faiss.normalize_L2(vectors)
        ids = np.arange(len(embeddings))

        index.add_with_ids(vectors, ids)
        faiss.write_index(index, FAISS_FILE)
        print("Saved FAISS index to disk")

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


    # -----------------------------
    # Guardrail Layer 1: Hardcoded
    # -----------------------------
    hardcoded_result = hardcoded_guardrail_check(user_question)
    if hardcoded_result != "benign":
        return {
            "answer": BLOCKED_RESPONSES[hardcoded_result],
            "citations": []
        }

    # -----------------------------
    # Guardrail Layer 2: LLM Classifier
    # -----------------------------
    intent = classify_intent(user_question, bedrock_runtime)
    if intent != "benign":
        return {
            "answer": BLOCKED_RESPONSES.get(intent, "I can’t help with that request."),
            "citations": []
        }
    
    # 1. Generate embedding for question
    payload = {
        "inputText": user_question,
        "embeddingTypes": ["float"]
    }
    response = bedrock_runtime.invoke_model(
        modelId=EMBEDDING_MODEL,
        body=json.dumps(payload),
        accept="application/json",
        contentType="application/json"
    )
    result = json.loads(response['body'].read())
    query_vector = np.array(result['embeddingsByType']['float']).astype("float32").reshape(1, -1)
    faiss.normalize_L2(query_vector)
    
    # 2. Retrieve top-k chunks from FAISS
    D, I = index.search(query_vector, top_k)
    SIM_THRESHOLD = 0.38

    # Pair scores with embedding IDs and filter
    filtered = [
        (score, embedding_id)
        for score, embedding_id in zip(D[0], I[0])
        if score >= SIM_THRESHOLD
    ]

    # If nothing is relevant enough, return early
    if not filtered:
        return {
            "answer": "Not found in provided documents.",
            "citations": []
        }

    retrieved_chunks = []
    citations = []

    for score, embedding_id in filtered:
        metadata = embeddings[int(embedding_id)]

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
    
    # 3. Build prompt for Claude
    prompt = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [
            {
                "role": "user",
                "content": (
                    "You are a factual question-answering assistant working for Dohtem.\n"
                    "Dohtem is a large company and maintains internal documents from vendors and third-party suppliers.\n\n"

                    "STRICT RULES:\n"
                    "1. Use ONLY the information explicitly stated in the provided context.\n"
                    "2. Do NOT use prior knowledge, assumptions, or general world knowledge.\n"
                    "3. If the answer cannot be fully answered using the context, reply with EXACTLY:\n"
                    "'Not found in provided documents.'\n"
                    "4. Do NOT add explanations, summaries, or partial answers when the answer is not found.\n\n"
                    "Context:\n"
                    f"{chr(10).join(retrieved_chunks)}\n\n"
                    "Question:\n"
                    f"{user_question}\n\n"
                    "Answer in 3–5 sentences max."
                )
            }
        ],
        "max_tokens": 300
    }
    
    #4.Invoke Claude
    response = bedrock_runtime.invoke_model(
        modelId=LLM_MODEL,
        body=json.dumps(prompt),
        accept="application/json",
        contentType="application/json"
    )
    answer = json.loads(response['body'].read())
    
    answer_text = answer['content'][0]['text']

    # Remove "Answer:" prefix
    answer_text = answer_text.replace("Answer:", "").split("Sources:")[0].strip()

    if answer_text.lower().startswith("not found"):
        citations = []


    return {
        "answer": answer_text,
        "citations": citations
    }

