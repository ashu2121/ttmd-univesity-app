import pandas as pd
import json
import faiss
import numpy as np
from openai import OpenAI
import os
import time
import datetime

# Set OpenAI key from environment
client = OpenAI(api_key=os.getenv("CLIENT_OPENAI_API_KEY"))

# CSV File paths
TRACK_FILE = "upload_records.json"
STATUS_FILE = "processing_status.json"

# JSON File paths
QA_TRACK_FILE = "qa_upload_records.json"
QA_STATUS_FILE = "qa_processing_status.json"

# Timestamp for unique filenames
timestamp = time.strftime("%Y%m%d%H%M%S")

# Filenames for CSV data
csv_json_file = f"csv_metadata_{timestamp}.json"
csv_faiss_file = f"csv_index_{timestamp}.faiss"

# Filenames for JSON data
json_json_file = f"json_metadata_{timestamp}.json"
json_faiss_file = f"json_index_{timestamp}.faiss"

print("=================================================================================")
print("CSV JSON file name >", csv_json_file)
print("CSV FAISS file name >", csv_faiss_file)
print("JSON JSON file name >", json_json_file)
print("JSON FAISS file name >", json_faiss_file)
print("=================================================================================")

# Update the processing status file
def update_status(status, error_message=""):
    status_record = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": status,
        "error": error_message
    }
    with open(STATUS_FILE, "w") as f:
        json.dump(status_record, f, indent=2)

update_status("processing")


# Update the processing status file
def update_qa_status(status, error_message=""):
    status_record = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": status,
        "error": error_message
    }
    with open(QA_STATUS_FILE, "w") as f:
        json.dump(status_record, f, indent=2)


update_qa_status("processing")


# ==================== PROCESS CSV DATA ====================
chunks_csv = []
successful_chunks_csv = []

try:
    if os.path.exists(TRACK_FILE):
        with open(TRACK_FILE, "r") as f:
            records = json.load(f)
            if records:
                last_file = records[-1]["saved_as"]
            else:
                raise FileNotFoundError("No uploaded CSV files found.")

    print(f"Processing CSV file: {last_file}")

    # Read CSV
    df = pd.read_csv(last_file, delimiter=',')
    df = df[df["college_name"].notnull() & df["course"].notnull() & df["cutoff"].notnull() & df["category"].notnull() & df["fee"].notnull()]
    df = df[df["exam"] == "NEET-MDS"].reset_index(drop=True)

    for _, row in df.iterrows():
        chunk = {
            "text": f"{row['college_name']} offers {row['course']} under {row['quota']} quota for {row['category']} category with a cutoff rank of {row['cutoff']}. Fee: ₹{row['fee']}. Type: {row['type']}. Counseling by {row['counseling_authority']}. Minority: {row['minority']}",
            "source": "CSV"
        }
        chunks_csv.append(chunk)

except Exception as e:
    update_status("failed", f"CSV Processing Error: {e}")
    print(f"Error processing CSV: {e}")

# Embed CSV Data
embedding_dim = 1536
csv_index = faiss.IndexFlatL2(embedding_dim)

try:
    for chunk in chunks_csv:
        if isinstance(chunk, dict) and "text" in chunk:
            content = chunk["text"].strip()
            response = client.embeddings.create(input=[content], model="text-embedding-3-small")
            embedding = response.data[0].embedding
            csv_index.add(np.array([embedding], dtype="float32"))
            successful_chunks_csv.append(chunk)

    with open(csv_json_file, "w", encoding="utf-8") as f:
        json.dump(successful_chunks_csv, f, ensure_ascii=False, indent=2)

    faiss.write_index(csv_index, csv_faiss_file)
    update_status("success")

    print("=================================================================================")
    print("CSV JSON file name >", csv_json_file)
    print("CSV FAISS file name >", csv_faiss_file)
    print("CSV data processed and saved successfully!")
    print("=================================================================================")

except Exception as e:
    update_status("failed", f"CSV Embedding Error: {e}")
    print(f"Error embedding CSV data: {e}")


# ==================== PROCESS JSON DATA ====================
chunks_json = []
successful_chunks_json = []

try:
    if os.path.exists(QA_TRACK_FILE):
        with open(QA_TRACK_FILE, "r") as f:
            records = json.load(f)
            if records:
                qa_last_file = records[-1]["saved_as"]
            else:
                raise FileNotFoundError("No uploaded JSON files found.")

    print(f"Processing JSON file: {qa_last_file}")

    # Read JSON
    if os.path.exists(qa_last_file):
        with open(qa_last_file, "r", encoding="utf-8") as f:
            new_entries = json.load(f)
            for entry in new_entries:
                if isinstance(entry, dict) and "question" in entry and "answer" in entry:
                    question = entry["question"].strip()
                    answer = entry["answer"].strip()
                    chunks_json.append({
                        "text": f"Question: {question} Answer: {answer}",
                        "source": "JSON"
                    })

except Exception as e:
    update_qa_status("failed", f"JSON Processing Error: {e}")
    print(f"Error processing JSON: {e}")

# Embed JSON Data
json_index = faiss.IndexFlatL2(embedding_dim)

try:
    for chunk in chunks_json:
        if isinstance(chunk, dict) and "text" in chunk:
            content = chunk["text"].strip()
            response = client.embeddings.create(input=[content], model="text-embedding-3-small")
            embedding = response.data[0].embedding
            json_index.add(np.array([embedding], dtype="float32"))
            successful_chunks_json.append(chunk)

    with open(json_json_file, "w", encoding="utf-8") as f:
        json.dump(successful_chunks_json, f, ensure_ascii=False, indent=2)

    faiss.write_index(json_index, json_faiss_file)
    update_qa_status("success")
 
    print("=================================================================================")
    print("JSON JSON file name >", json_json_file)
    print("JSON FAISS file name >", json_faiss_file)
    print("JSON data processed and saved successfully!")
    print("=================================================================================")

    with open("current_files.json", "w") as f:
        json.dump({"json_json_file": json_json_file, "json_faiss_file": json_faiss_file,
                    "csv_json_file": csv_json_file, "csv_faiss_file": csv_faiss_file}, f)


except Exception as e:
    update_qa_status("failed", f"JSON Embedding Error: {e}")
    print(f"Error embedding JSON data: {e}")
