from flask import Flask, request, jsonify
from openai import OpenAI
import faiss
import numpy as np
import json
import os
from flask_cors import CORS
import datetime

#if not os.path.exists("university_index.faiss"):
#    import prepare_data_faiss  # Will auto-run and generate both files

with open(f"current_files.json", "r") as f:
    file_config = json.load(f)
json_file1 = file_config["json_file"]
faiss_file1 = file_config["faiss_file"]
print(json_file1)

# Load metadata and FAISS index
with open(json_file1, "r", encoding="utf-8") as f:
    metadata = json.load(f)

index = faiss.read_index(faiss_file1)
embedding_dim = 1536

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# OpenAI client setup
#client = OpenAI(api_key=os.environ.)

# Flask app setup
app = Flask(__name__)
CORS(app)  # Enable for all origins
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to get embedding for a query
def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

# Function to search for relevant chunks
def search_chunks(query, k=5):
    query_vector = np.array([get_embedding(query)], dtype="float32")
    distances, indices = index.search(query_vector, k)
    return [metadata[i] for i in indices[0]]

# Function to generate final GPT answer
def ask_gpt_with_context(query):
    context = "\n".join(search_chunks(query))
    messages = [
        {"role": "system", "content": "You are a helpful university admission assistant. Answer only from the provided context."},
        {"role": "user", "content": f"Data:\n{context}\n\nQuestion: {query}"}
    ]
    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=messages
    )
    return response.choices[0].message.content



@app.route("/ask1", methods=["GET"])
def ask1():
    query = request.args.get("query", "").strip()
    if not query:
        return jsonify({"error": "Missing query"}), 400
    try:
        answer = ask_gpt_with_context(query)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# POST endpoint
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "Missing query"}), 400
    try:
        answer = ask_gpt_with_context(query)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500



UPLOAD_FOLDER = "uploads"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
TRACK_FILE = "upload_records.json"
# Initialize tracking file if not exists
if not os.path.exists(TRACK_FILE):
    with open(TRACK_FILE, "w") as f:
        f.write("[]")

@app.route("/upload", methods=["POST"])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 408

        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "No selected file"}), 410
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
        filepath = os.path.join(UPLOAD_FOLDER, timestamp+".csv")
        file.save(filepath)

        record = {
            "timestamp": timestamp,
            "filename": file.filename,
            "saved_as": filepath
        }

        # Append to the tracking file
        with open(TRACK_FILE, "r+") as f:
            records = json.load(f)
            records.append(record)
            f.seek(0)
            json.dump(records, f, indent=2)

        return jsonify({"status": "success", "filepath": filepath}), 200

    except Exception as e:
        print(str(e))
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)