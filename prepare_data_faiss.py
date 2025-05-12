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

# File paths
TRACK_FILE = "upload_records.json"
STATUS_FILE = "processing_status.json"
timestamp = time.strftime("%Y%m%d%H%M%S")
json_file = f"university_metadata_{timestamp}.json"
faiss_file = f"university_index_{timestamp}.faiss"


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

# Get the last saved file
try:
    if os.path.exists(TRACK_FILE):
        with open(TRACK_FILE, "r") as f:
            records = json.load(f)
            if records:
                last_file = records[-1]["saved_as"]
            else:
                raise FileNotFoundError("No uploaded files found.")
    else:
        raise FileNotFoundError("Upload record file not found.")

    print(f"Processing file: {last_file}")

except Exception as e:
    update_status("failed", str(e))
    print(f"Error initializing file processing: {e}")
    exit()

# Step 1: Read CSV with correct delimiter
try:
    df = pd.read_csv(last_file, delimiter=',')
    print("🧾 CSV Headers:", df.columns.tolist())

    # Filter usable rows
    df = df[df["college_name"].notnull() & df["course"].notnull() & df["cutoff"].notnull() & df["category"].notnull() & df["fee"].notnull()]
    print(f"✅ Loaded {len(df)} valid rows.")

    df = df[df["exam"] == "NEET-MDS"].reset_index(drop=True)

except Exception as e:
    update_status("failed", f"CSV Reading Error: {e}")
    print(f"Error reading CSV: {e}")
    exit()

# Step 2: Create JSON Chunks
chunks = []
try:
    for _, row in df.iterrows():
        university = row["college_name"]
        course = row["course"]
        quota = row["quota"]
        category = row["category"]
        cutoff = row["cutoff"]
        fee = row["fee"]
        round_ = row["round"]
        authority = row["counseling_authority"]
        utype = row["type"]
        minority = row["minority"] if pd.notna(row["minority"]) else "None"

        chunk = (
            f"{university} offers {course} under {quota} quota for {category} category in round {round_} "
            f"with a cutoff rank of {cutoff}. Fee: ₹{fee}. College Type: {utype}. Counseling by {authority}. Minority: {minority}"
        )
        chunks.append(chunk)

    # Save chunks to JSON
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

except Exception as e:
    update_status("failed", f"Data Processing Error: {e}")
    print(f"Error creating JSON chunks: {e}")
    exit()

# Step 3: Embed chunks and save FAISS index
embedding_dim = 1536
index = faiss.IndexFlatL2(embedding_dim)

try:
    for chunk in chunks:
        response = client.embeddings.create(input=[chunk], model="text-embedding-3-small")
        embedding = response.data[0].embedding
        index.add(np.array([embedding], dtype="float32"))

    faiss.write_index(index, faiss_file)
    print("✅ JSON and FAISS files created.")
    update_status("success")

    # Save filenames to a config file
    with open("current_files.json", "w") as f:
        json.dump({"json_file": json_file, "faiss_file": faiss_file}, f)


except Exception as e:
    update_status("failed", f"Embedding/FAISS Error: {e}")
    print(f"Error during FAISS processing: {e}")
