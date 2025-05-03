import pandas as pd
import json
import faiss
import numpy as np
from openai import OpenAI
import os

# Set OpenAI key from environment
client = OpenAI(api_key="")
# File paths
csv_file = "Improved_NEET_MDS_Dataset.csv"
json_file = "university_metadata.json"
faiss_file = "university_index.faiss"

# Step 1: Read CSV
df = pd.read_csv(csv_file, delimiter=';')

# Filter usable rows
df = df[df["college_name"].notnull() & df["course"].notnull() & df["cutoff"].notnull() & df["category"].notnull() & df["fee"].notnull()]
print("🧾 CSV Columns:", df.columns.tolist())
print(f"✅ Loaded {len(df)} valid rows.")

df = df[df["exam"] == "NEET-MDS"].reset_index(drop=True)

# Step 2: Create JSON Chunks
chunks = []
for _, row in df.iterrows():
    university = row["college_name"]
    course = row["course"]
    quota = row["quota"]
    category = row["category"]
    print("cutoff > ", row['cutoff'])
    continue
    cutoff = 100
    fee = row["fee"]
    round_ = row["round"]
    authority = row["counseling_authority"]
    utype = row["type"]
    state = row["satte"]
    minority = row["minority"] if pd.notna(row["minority"]) else "None"

    chunk = (
        f"{university} offers {course} under {quota} quota for {category} category in round {round_} "
        f"with a cutoff rank of {cutoff}. Fee: ₹{fee}. Type: {utype}. Counseling by {authority}. Minority: {minority}. State: {state}."
    )
    chunks.append(chunk)

# Save chunks to JSON
with open(json_file, "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)

# Step 3: Embed chunks and save FAISS index
embedding_dim = 1536
index = faiss.IndexFlatL2(embedding_dim)

for chunk in chunks:
    response = client.embeddings.create(input=[chunk], model="text-embedding-3-small")
    embedding = response.data[0].embedding
    index.add(np.array([embedding], dtype="float32"))

faiss.write_index(index, faiss_file)
print("✅ JSON and FAISS files created.")
