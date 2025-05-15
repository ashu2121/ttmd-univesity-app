from flask import Flask, request, jsonify
from openai import OpenAI
import faiss
import numpy as np
import json
import os
from flask_cors import CORS
import datetime
import threading
import subprocess
import traceback

# --- Global Configuration & Initialization ---

embedding_dim = 1536
client = OpenAI(api_key=os.getenv("CLIENT_OPENAI_API_KEY"))

# Flask app setup
app = Flask(__name__)
CORS(app)  # Enable for all origins
# CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True) # More specific if needed

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

QA_UPLOAD_FOLDER = 'qa_uploads'
if not os.path.exists(QA_UPLOAD_FOLDER):
    os.makedirs(QA_UPLOAD_FOLDER)
# app.config['QA_UPLOAD_FOLDER'] = QA_UPLOAD_FOLDER # If needed by Flask

# Function to get embedding for a query
def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

# Load FAISS indices and metadata
try:
    with open("current_files.json", "r") as f:
        file_config = json.load(f)
        csv_json_file = file_config.get("csv_json_file")
        csv_faiss_file = file_config.get("csv_faiss_file")
        json_json_file = file_config.get("json_json_file")
        json_faiss_file = file_config.get("json_faiss_file")
except Exception as e:
    print(f"Error loading current_files.json: {e}. Indices and metadata might not be loaded.")
    csv_json_file, csv_faiss_file, json_json_file, json_faiss_file = None, None, None, None

csv_index, json_index = None, None
csv_metadata, json_metadata = [], []

if csv_faiss_file and os.path.exists(csv_faiss_file):
    csv_index = faiss.read_index(csv_faiss_file)
    if csv_json_file and os.path.exists(csv_json_file):
        with open(csv_json_file, "r", encoding="utf-8") as f:
            csv_metadata = json.load(f)
    else: print(f"Warning: CSV metadata file {csv_json_file} not found.")
if json_faiss_file and os.path.exists(json_faiss_file):
    json_index = faiss.read_index(json_faiss_file)
    if json_json_file and os.path.exists(json_json_file):
        with open(json_json_file, "r", encoding="utf-8") as f:
            json_metadata = json.load(f)
    else: print(f"Warning: JSON metadata file {json_json_file} not found.")

def remove_substring(main_string, start_string, end_string):
    start_index = main_string.find(start_string)
    if start_index == -1:
        return main_string  # Start string not found

    end_index = main_string.find(end_string, start_index + len(start_string))
    if end_index == -1:
        return main_string  # End string not found

    return main_string[:start_index] + main_string[end_index + len(end_string):]

# This function is defined but not currently used. Could be for more advanced scoring.
def calculate_combined_score(dist, cutoff_rank, user_rank):
    WEIGHT_EMBEDDING = 0.7
    WEIGHT_RANK = 0.3

    # Embedding-based score
    embedding_score = 1 / (dist + 1e-5)

    # Rank-based score
    if user_rank is not None:
        rank_score = 1 / (abs(cutoff_rank - user_rank) + 1e-5)
    else:
        rank_score = 1.0  # Neutral for general queries

    # Combined score
    combined_score = (embedding_score * WEIGHT_EMBEDDING) + (rank_score * WEIGHT_RANK)
    return combined_score

@app.route("/ask1", methods=["GET"])
def ask1():
    query = request.args.get("query", "").strip()
    if not query:
        return jsonify({"error": "Missing query parameter"}), 400

    user_rank = None
    try:
        # Basic rank extraction: assumes rank is a number in the query
        for word in query.lower().split():
            if word.isdigit():
                user_rank = int(word)
                break # Take the first number found as rank
    except ValueError:
        pass # user_rank remains None

    try:
        query_embedding = get_embedding(query)
        query_vector = np.array([query_embedding], dtype="float32")
    except Exception as e:
        print(f"Error embedding query: {e}")
        return jsonify({"error": "Failed to embed query."}), 500

    k_faiss = 10  # Number of results to fetch from FAISS before filtering
    all_results = []

    # Search CSV Data
    if csv_index and csv_metadata:
        try:
            distances, indices = csv_index.search(query_vector, k_faiss)
            for i in range(len(indices[0])):
                idx = indices[0][i]
                dist = distances[0][i]
                if not (0 <= idx < len(csv_metadata)): continue

                entry = csv_metadata[idx]
                entry_cutoff_rank = entry.get("cutoff_rank", float("inf"))

                # Rank Filter: Include if user_rank not specified, or if entry is general (inf cutoff),
                # or if user's rank is good enough (user_rank <= entry_cutoff_rank)
                if user_rank is None or entry_cutoff_rank == float("inf") or user_rank <= entry_cutoff_rank:
                    score = 1 / (dist + 1e-5)
                    all_results.append({
                        "text": entry["text"],
                        "score": score,
                        "cutoff_rank": entry_cutoff_rank,
                        "source": "CSV"
                    })
        except Exception as e:
            print(f"Error searching CSV index: {e}")
            # Potentially return partial results or an error

    # Search JSON Data
    if json_index and json_metadata:
        try:
            distances, indices = json_index.search(query_vector, k_faiss)
            for i in range(len(indices[0])):
                idx = indices[0][i]
                dist = distances[0][i]
                if not (0 <= idx < len(json_metadata)): continue

                entry = json_metadata[idx]
                # JSON entries might not have cutoff_rank or it's irrelevant
                entry_cutoff_rank = entry.get("cutoff_rank", float("inf"))

                # Rank filter (less strict for JSON, usually passes if user_rank is None or cutoff is inf)
                if user_rank is None or entry_cutoff_rank == float("inf") or user_rank <= entry_cutoff_rank:
                    score = 1 / (dist + 1e-5)
                    result_text = remove_substring(entry["text"], "Question: ", "? Answer:")
                    all_results.append({
                        "text": result_text,
                        "score": score,
                        "cutoff_rank": entry_cutoff_rank, # Usually inf for JSON
                        "source": "JSON"
                    })
        except Exception as e:
            print(f"Error searching JSON index: {e}")

    if not all_results:
        if not csv_index and not json_index:
             return jsonify({"answer": "Backend data sources (FAISS indices) are not loaded. Please check server setup."}), 503
        return jsonify({"answer": "No relevant information found matching your criteria."}), 404

    # Sort combined results by score
    sorted_results = sorted(all_results, key=lambda x: x["score"], reverse=True)

    return jsonify({"answer": sorted_results[0]["text"]}), 200

# --- Helper for /ask endpoint ---
def search_and_filter_chunks(query, user_rank, k=5):
    try:
        query_embedding = get_embedding(query)
        query_vector = np.array([query_embedding], dtype="float32")
    except Exception as e:
        print(f"Error embedding query for /ask: {e}")
        return [] # Return empty list on embedding failure

    # Use k_faiss for initial FAISS fetch, then k for final top results
    k_faiss = max(k * 2, 10) # Fetch more from FAISS to have enough after filtering
    
    # This part is very similar to /ask1, could be refactored into a common function
    # For brevity, repeating the search logic here.
    # In a larger app, abstract this search & filter logic.
    
    retrieved_texts = []
    # Search CSV
    if csv_index and csv_metadata:
        distances, indices = csv_index.search(query_vector, k_faiss)
        for i in range(len(indices[0])):
            idx, dist = indices[0][i], distances[0][i]
            if 0 <= idx < len(csv_metadata):
                entry = csv_metadata[idx]
                entry_cutoff_rank = entry.get("cutoff_rank", float("inf"))
                if user_rank is None or entry_cutoff_rank == float("inf") or user_rank <= entry_cutoff_rank:
                    retrieved_texts.append(entry["text"]) # Could also add score for sorting if needed
    # Search JSON
    if json_index and json_metadata:
        distances, indices = json_index.search(query_vector, k_faiss)
        for i in range(len(indices[0])):
            idx, dist = indices[0][i], distances[0][i]
            if 0 <= idx < len(json_metadata):
                entry = json_metadata[idx]
                # Assuming JSON doesn't need strict rank filtering or has 'inf' cutoff
                retrieved_texts.append(remove_substring(entry["text"], "Question: ", "? Answer:"))

    # For now, just return a list of texts. Could be more sophisticated with scoring and selection.
    return list(set(retrieved_texts))[:k] # Return unique texts, up to k

# POST endpoint
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "Missing query"}), 400

    user_rank = None # Extract rank if needed for context filtering
    try:
        for word in query.lower().split():
            if word.isdigit():
                user_rank = int(word)
                break
    except ValueError:
        pass

    try:
        # Get relevant chunks from both CSV and JSON, applying rank filtering
        context_chunks = search_and_filter_chunks(query, user_rank, k=3) # Get top 3 diverse chunks

        if not context_chunks:
            if not csv_index and not json_index:
                 return jsonify({"answer": "Backend data sources (FAISS indices) are not loaded. Please check server setup."}), 503
            return jsonify({"answer": "I couldn't find specific information for your query in the available data."})

        context_str = "\n\n---\n\n".join(context_chunks)
        
        messages = [
            {"role": "system", "content": "You are a helpful university admission assistant. Answer the user's question based *only* on the following provided context. Do not use any external knowledge or make assumptions. If the context does not contain the answer, say that you cannot find the information in the provided data."},
            {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {query}"}
        ]
        response = client.chat.completions.create(model="gpt-4o-mini-2024-07-18", messages=messages)
        answer = response.choices[0].message.content
        return jsonify({"answer": answer})
    except Exception as e:
        print(f"Error in /ask endpoint: {e}\n{traceback.format_exc()}")
        return jsonify({"error": "An internal error occurred while processing your request."}), 500

TRACK_FILE = "upload_records.json"
# Initialize tracking file if not exists

if not os.path.exists(TRACK_FILE):
    with open(TRACK_FILE, "w") as f:
        f.write("[]")

def run_faiss_processing():
    try:
        print("Starting FAISS processing...")
        subprocess.Popen(["python", "prepare_data_faiss.py"])
    except Exception as e:
        print(f"Error running FAISS processing: {e}")

@app.route("/upload", methods=["POST"])
def upload_file():
    try:
        # Check processing status
        if os.path.exists("processing_status.json"):
            with open("processing_status.json", "r") as f:
                status_data = json.load(f)
                if status_data.get("status") == "processing":
                    return jsonify({"error": "Previous file processing is still in progress."}), 503 # Service Unavailable
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

        # Start FAISS processing in a new thread
        threading.Thread(target=run_faiss_processing).start()

        return jsonify({"status": "success", "filepath": filepath}), 200
        
    except Exception as e:
        print(f"Error in /upload endpoint: {e}\n{traceback.format_exc()}")
        return jsonify({"error": "An internal error occurred during file upload."}), 500

QA_TRACK_FILE = "qa_upload_records.json"
if not os.path.exists(QA_TRACK_FILE):
    with open(QA_TRACK_FILE, "w") as f:
        f.write("[]")

@app.route("/upload_qa", methods=["POST"])
def upload_question_answer_file():
    try:
        if os.path.exists("qa_processing_status.json"):
            with open("qa_processing_status.json", "r") as f:
                status_data = json.load(f)
                if status_data.get("status") == "processing":
                    return jsonify({"error": "Previous Q&A file processing is still in progress."}), 503

        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 408

        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "No selected file"}), 410
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
        filepath = os.path.join(QA_UPLOAD_FOLDER, timestamp+".json")
        file.save(filepath)

        record = {
            "timestamp": timestamp,
            "filename": file.filename,
            "saved_as": filepath
        }

        # Append to the tracking file
        with open(QA_TRACK_FILE, "r+") as f:
            records = json.load(f)
            records.append(record)
            f.seek(0)
            json.dump(records, f, indent=2)

        threading.Thread(target=run_faiss_processing).start()

        return jsonify({"status": "success", "filepath": filepath}), 200
        
    except Exception as e:
        print(f"Error in /upload_qa endpoint: {e}\n{traceback.format_exc()}")
        return jsonify({"error": "An internal error occurred during Q&A file upload."}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)