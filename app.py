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

#if not os.path.exists("university_index.faiss"):
#    import prepare_data_faiss  # Will auto-run and generate both files

with open(f"current_files.json", "r") as f:
    file_config = json.load(f)
#json_file1 = file_config["csv_json_file"]
#faiss_file1 = file_config["csv_faiss_file"]
#print(json_file1)

# Load metadata and FAISS index
#with open(json_file1, "r", encoding="utf-8") as f:
#    metadata = json.load(f)

#index = faiss.read_index(faiss_file1)
embedding_dim = 1536

client = OpenAI(api_key=os.getenv("CLIENT_OPENAI_API_KEY"))


# OpenAI client setup
#client = OpenAI(api_key=os.environ.)

# Flask app setup
app = Flask(__name__)
CORS(app)  # Enable for all origins
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

QA_UPLOAD_FOLDER = 'qa_uploads'
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


try:
    with open("current_files.json", "r") as f:
        file_config = json.load(f)
        csv_json_file = file_config.get("csv_json_file")
        csv_faiss_file = file_config.get("csv_faiss_file")
        json_json_file = file_config.get("json_json_file")
        json_faiss_file = file_config.get("json_faiss_file")
except Exception as e:
    csv_json_file, csv_faiss_file, json_json_file, json_faiss_file = None, None, None, None
    print(f"Error loading current files: {e}")

# Load FAISS indices
csv_index, json_index = None, None
if csv_faiss_file and os.path.exists(csv_faiss_file):
    csv_index = faiss.read_index(csv_faiss_file)
if json_faiss_file and os.path.exists(json_faiss_file):
    json_index = faiss.read_index(json_faiss_file)

# Load Metadata
csv_metadata, json_metadata = [], []
if csv_json_file and os.path.exists(csv_json_file):
    with open(csv_json_file, "r", encoding="utf-8") as f:
        csv_metadata = json.load(f)

if json_json_file and os.path.exists(json_json_file):
    with open(json_json_file, "r", encoding="utf-8") as f:
        json_metadata = json.load(f)

from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)


def remove_substring(main_string, start_string, end_string):
    start_index = main_string.find(start_string)
    if start_index == -1:
        return main_string  # Start string not found

    end_index = main_string.find(end_string, start_index + len(start_string))
    if end_index == -1:
        return main_string  # End string not found

    return main_string[:start_index] + main_string[end_index + len(end_string):]


@app.route("/ask1", methods=["GET"])
def ask1():
    isAnswerFromJsonFileOnly = False
    query = request.args.get("query", "").strip()
    if not query:
        return jsonify({"error": "Missing query parameter"}), 400

    try:
        if not csv_index or not json_index:
            return jsonify({"error": "FAISS indices not loaded."}), 500

        # Extract rank and category from query
        rank, category = None, None
        try:
            for word in query.lower().split():
                if word.isdigit():
                    rank = int(word)
                elif word in ["sc", "st", "general", "minority"]:
                    category = word.upper()
        except ValueError:
            pass

        # Embed the query
        try:
            response = client.embeddings.create(input=[query], model="text-embedding-3-small")
            query_vector = np.array([response.data[0].embedding], dtype="float32")
            print("Query Vector:", query_vector)
        except Exception as e:
            print(f"Error embedding query: {e}")
            return jsonify({"error": "Failed to embed query."}), 500

        # Search CSV Index
        csv_results = []
        if csv_index:
            csv_distances, csv_indices = csv_index.search(query_vector, 5)
            for dist, idx in zip(csv_distances[0], csv_indices[0]):
                if idx < len(csv_metadata):
                    entry = csv_metadata[idx]
                    cutoff_rank = entry.get("cutoff_rank", float("inf"))
                    entry_category = entry.get("category")  # No default category assigned
                    score = 1 / (dist + 1e-5)

                    # Handling queries without rank
                    if rank is None:
                        csv_results.append({"text": entry["text"], "score": score, "source": "CSV"})
                    # Include only if rank is higher than the cutoff rank and category matches
                    elif rank <= cutoff_rank and (category is None or category == entry_category):
                        csv_results.append({"text": entry["text"], "score": score, "source": "CSV"})

        # Search JSON Index
        json_results = []
        if json_index:
            json_distances, json_indices = json_index.search(query_vector, 5)
            for dist, idx in zip(json_distances[0], json_indices[0]):
                if idx < len(json_metadata):
                    entry = json_metadata[idx]
                    cutoff_rank = entry.get("cutoff_rank", float("inf"))
                    entry_category = entry.get("category")  # No default category assigned
                    score = 1 / (dist + 1e-5)

                    # Handling queries without rank
                    if rank is None:
                        result = remove_substring(entry["text"], "Question: ", "? Answer:")
                        json_results.append({"text": result, "score": score, "source": "JSON"})
                    # Include only if rank is higher than the cutoff rank and category matches
                    elif rank <= cutoff_rank and (category is None or category == entry_category):
                        result = remove_substring(entry["text"], "Question: ", "? Answer:")
                        json_results.append({"text": result, "score": score, "source": "JSON"})

        # Additional General Query Handling
        if rank is None and not csv_results and not json_results:
            for dist, idx in zip(csv_distances[0], csv_indices[0]):
                if idx < len(csv_metadata):
                    entry = csv_metadata[idx]
                    score = 1 / (dist + 1e-5)
                    csv_results.append({"text": entry["text"], "score": score, "source": "CSV"})

            for dist, idx in zip(json_distances[0], json_indices[0]):
                if idx < len(json_metadata):
                    entry = json_metadata[idx]
                    score = 1 / (dist + 1e-5)
                    result = remove_substring(entry["text"], "Question: ", "? Answer:")
                    json_results.append({"text": result, "score": score, "source": "JSON"})

        # Debugging Output
        print("CSV Results:", csv_results)
        print("JSON Results:", json_results)

        # Combine and Rank Results
        combined_results = csv_results + json_results
        combined_results = sorted(combined_results, key=lambda x: x["score"], reverse=True)

        # Display Only the Highest Scoring Response
        #top_response = combined_results[0]["text"] 
        top_response = [result["text"] for result in combined_results[:5]]
        print("=================================")
        print("\n" .join(top_response))
        print("=================================")
 
        finalResponse =  str("\n" .join(top_response))
        top_response = combined_results if combined_results else "No relevant information found."

        return jsonify({"answer":  finalResponse }), 200

    except Exception as e:
        print(f"Error in /ask1 endpoint: {e}")
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


def run_faiss_processing():
    try:
        print("Starting FAISS processing...")
        subprocess.Popen(["python", "prepare_data_faiss.py"])
    except Exception as e:
        print(f"Error running FAISS processing: {e}")


@app.route("/upload", methods=["POST"])
def upload_file():
    try:
        with open(f"processing_status.json", "r") as f:
            file_config = json.load(f)
        previousFileProcessingStatus = file_config["status"]
        if(previousFileProcessingStatus == "processing"):
            return jsonify({"error": "previous file still in process"}), 510
        


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

        # initi
        print(jsonify({"status": "success", "filepath": filepath})), 200
        threading.Thread(target=run_faiss_processing).start()

        return jsonify({"status": "success", "filepath": filepath}), 200
        
    except Exception as e:
        print(str(e))
        return jsonify({"error": str(e)}), 500


QA_TRACK_FILE = "qa_upload_records.json"
if not os.path.exists(QA_TRACK_FILE):
    with open(QA_TRACK_FILE, "w") as f:
        f.write("[]")

@app.route("/upload_qa", methods=["POST"])
def upload_question_answer_file():
    try:
        with open(f"qa_processing_status.json", "r") as f:
            file_config = json.load(f)
        
        previousFileProcessingStatus = file_config["status"]
        if(previousFileProcessingStatus == "processing"):
            return jsonify({"error": "previous file still in process"}), 510
    

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

        # initi
        print(jsonify({"status": "success", "filepath": filepath})), 200
        threading.Thread(target=run_faiss_processing).start()

        return jsonify({"status": "success", "filepath": filepath}), 200
        
    except Exception as e:
        print(str(e))
        return jsonify({"error": str(e)}), 500




if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)