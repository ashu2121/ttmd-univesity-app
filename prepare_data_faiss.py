import pandas as pd
import json
import faiss
import numpy as np
from openai import OpenAI
import os
import time
import datetime
import re

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
        # More robustly parse cutoff_rank to a numeric value
        raw_cutoff_str = str(row['cutoff']).strip()
        # Remove commas that might be used as thousands separators
        cleaned_str_for_num_extraction = raw_cutoff_str.replace(',', '')
        
        # Search for the first sequence of digits (can include a decimal, though ranks are usually integers)
        # This will find numbers like "9868", "12000.0", even if embedded like "Approx 9868"
        match = re.search(r'\d+(\.\d+)?', cleaned_str_for_num_extraction)
        
        if match:
            try:
                cutoff_val = float(match.group(0))
            except ValueError: # Should be rare if regex matches a number-like string
                cutoff_val = 0
        else:
            cutoff_val = 0 # No numeric part found

        # The text will still display the original cutoff string from the CSV
        chunk = {
            "text": f"{row['college_name']} offers {row['course']} under {row['quota']} quota for {row['category']} category with a cutoff rank of {row['cutoff']}. Fee: ₹{row['fee']}. Type: {row['type']}. Counseling by {row['counseling_authority']}. Minority: {row['minority']}",
            "source": "CSV",
            "cutoff_rank": cutoff_val  # Store numeric cutoff rank
        }
        chunks_csv.append(chunk)

except Exception as e:
    update_status("failed", f"CSV Processing Error: {e}")
    print(f"Error processing CSV: {e}")



def generate_query_vector(query, rank=None):
    # Embed the query
    response = client.embeddings.create(input=[query], model="text-embedding-3-small")
    embedding = np.array(response.data[0].embedding, dtype="float32")

    # Assign rank or neutral value (float("inf")) for general queries
    rank_value = rank if rank is not None else float("inf")
    query_vector = np.append(embedding, rank_value).reshape(1, -1)
    
    return query_vector

# Embed CSV Data
embedding_dim = 1536
new_dim = embedding_dim + 1  # Additional dimension for cutoff rank

csv_index = faiss.IndexFlatL2(embedding_dim)

try:
    embedding_dim = 1536  # Original embedding dimension
    csv_index = faiss.IndexFlatL2(embedding_dim)
    successful_chunks_csv = []

    for chunk in chunks_csv:
        if isinstance(chunk, dict) and "text" in chunk:
            content = chunk["text"].strip()

            # Embed the text
            try:
                response = client.embeddings.create(input=[content], model="text-embedding-3-small")
                embedding = np.array(response.data[0].embedding, dtype="float32")

                # Check for NaN or Infinity values
                if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
                    print(f"Warning: Invalid embedding detected for text: {content}")
                    continue  # Skip this entry

                # Add to FAISS index
                csv_index.add(np.array([embedding], dtype="float32"))

                # Add embedding to chunk data
                chunk["embedding"] = embedding.tolist()
                successful_chunks_csv.append(chunk)

            except Exception as e:
                print(f"Error generating embedding for text: {content} | Error: {e}")
                continue

    # Save CSV metadata with embeddings
    with open(csv_json_file, "w", encoding="utf-8") as f:
        json.dump(successful_chunks_csv, f, ensure_ascii=False, indent=2)

    # Save the FAISS index
    faiss.write_index(csv_index, csv_faiss_file)
    update_status("success")

    print(f"FAISS index (dimension: {csv_index.d}) and CSV JSON saved successfully.")
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
# Embed JSON Data
json_index = faiss.IndexFlatL2(embedding_dim)

try:
    for chunk in chunks_json:
        if isinstance(chunk, dict) and "text" in chunk:
            content = chunk["text"].strip()
            response = client.embeddings.create(input=[content], model="text-embedding-3-small")
            embedding = response.data[0].embedding

            # Add to FAISS index
            json_index.add(np.array([embedding], dtype="float32"))
            
            # Include the embedding in the chunk data
            chunk["embedding"] = embedding # embedding is already a list here
            successful_chunks_json.append(chunk)

    # Save JSON metadata with embeddings
    with open(json_json_file, "w", encoding="utf-8") as f:
        json.dump(successful_chunks_json, f, ensure_ascii=False, indent=2)

    # Save the FAISS index
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
