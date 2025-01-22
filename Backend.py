from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
import numpy as np
import faiss
import pickle

app = Flask(__name__)

# Load models and FAISS index
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
model_flan = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

faiss_index_path = 'thinkhat_faiss_index.index'
index = faiss.read_index(faiss_index_path)

with open('thinkhat_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

@app.route('/query', methods=['POST'])
def handle_query():
    data = request.json
    query = data.get('query', '')
    if not query:
        return jsonify({"error": "Query not provided"}), 400

    # Search FAISS
    query_embedding = embedding_model.encode(query, convert_to_numpy=True)
    query_embedding = np.expand_dims(query_embedding, axis=0)
    distances, indices = index.search(query_embedding, k=5)

    # Gather results
    results = []
    for idx in indices[0]:
        if idx < len(metadata):
            results.append(metadata[idx])

    if not results:
        return jsonify({"response": "No relevant information found."})

    # Clean and summarize results
    seen_contents = set()
    cleaned_results = []
    for result in results:
        if result['content'][:150] not in seen_contents:
            seen_contents.add(result['content'][:150])
            cleaned_results.append(result)

    for result in cleaned_results:
        content = result['content'][:1024]
        summary = summarizer(content, max_length=100, min_length=30, do_sample=False)
        result['summary'] = summary[0]['summary_text']

    # Generate response
    prompt = "Based on the following summaries, provide a detailed list of services thinkhat offers:\n\n"
    for i, result in enumerate(cleaned_results, 1):
        prompt += f"{i}. {result['summary']}\n"

    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).input_ids
    output = model_flan.generate(
        input_ids,
        max_length=200,
        num_beams=5,
        repetition_penalty=2.0,
        length_penalty=1.0,
        early_stopping=True
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)    
    return jsonify({"response": response, "results": cleaned_results})

if __name__ == '__main__':
    app.run(debug=False)