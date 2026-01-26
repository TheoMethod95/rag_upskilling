import requests

DELETE_API_URL = "http://localhost:8000/delete"
#DELETE_API_URL = "http://3.234.222.103:8000/delete"

def delete_and_update(file_keys: list):
    """
    Request deletion + reindex from the RAG API.
    """
    for key in file_keys:
        if not key.endswith(".txt") and not key.endswith(".pdf"):
            print(f"Skipping {key}: Only .txt and .pdf files are allowed")
            continue

        print(f"Requesting deletion of {key}...")
        resp = requests.post(
            DELETE_API_URL,
            params={"file_name": key}
        )

        if resp.status_code != 200:
            raise RuntimeError(resp.text)

    print("Deletion + RAG update complete!")
