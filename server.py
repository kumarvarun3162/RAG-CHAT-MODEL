import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, request, jsonify, send_from_directory
from src.retriever import load_retriever, retrieve, is_retrievable
from src.generator import answer_question

app = Flask(__name__, static_folder="static")

print("Loading model and vector DB...")
model, collection = load_retriever()
print("System ready.\n")


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/status")
def status():
    return jsonify({
        "chunks": collection.count(),
        "spec":   "TS 38.300 Release 19",
        "model":  "all-mpnet-base-v2",
        "llm":    "mistral"
    })


@app.route("/ask", methods=["POST"])
def ask():
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "No JSON body"}), 400

    query = body.get("query", "").strip()
    if not query:
        return jsonify({"error": "Empty query"}), 400

    retrieved = retrieve(query, model, collection, top_k=4)

    if not is_retrievable(retrieved):
        return jsonify({
            "answer":  "This topic is not covered in the provided 3GPP documentation.",
            "sources": [],
            "found":   False
        })

    result = answer_question(query, model, collection)

    return jsonify({
        "answer":  result["answer"],
        "sources": result["chunks"],
        "found":   True
    })


if __name__ == "__main__":
    print("Running at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)