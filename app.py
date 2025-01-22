import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
from datetime import datetime
import numpy as np
import faiss
import pickle

# Streamlit app configuration
st.set_page_config(page_title="Thinkhat Chatbot", page_icon="🤖", layout="wide")

# Add a header with custom styling
st.markdown(
    """
    <h1 style="text-align: center; color: #4CAF50;">🤖 Thinkhat Chatbot</h1>
    <p style="text-align: center; color: #888;">Your intelligent assistant for all your queries.</p>
    """,
    unsafe_allow_html=True,
)

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Function to add messages to the chat history
def add_to_chat(message, sender="user"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.chat_history.append({"sender": sender, "message": message, "timestamp": timestamp})

# Load models and FAISS index
@st.cache_resource
def load_models():
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
    model_flan = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return embedding_model, tokenizer, model_flan, summarizer

@st.cache_resource
def load_faiss_index():
    faiss_index_path = 'thinkhat_faiss_index.index'
    index = faiss.read_index(faiss_index_path)
    with open('thinkhat_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    return index, metadata

embedding_model, tokenizer, model_flan, summarizer = load_models()
index, metadata = load_faiss_index()

# Function to process the query
def process_query():
    query = st.session_state.temp_input  # Get the current input value
    if not query.strip():
        add_to_chat("⚠️ Please enter a query!", sender="bot")
        return

    add_to_chat(f"You: {query}", sender="user")
    add_to_chat("⏳ Processing query... Please wait.", sender="bot")

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
        add_to_chat("🤖 Thinkhat Bot: No relevant information found.", sender="bot")
        st.session_state.temp_input = ""  # Clear input
        return

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
    prompt = "Based on the following summaries, provide a detailed list of services Thinkhat offers:\n\n"
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
    add_to_chat(f"🤖 Thinkhat Bot: {response}", sender="bot")
    st.session_state.temp_input = ""  # Clear input

# Input box for user query with callback
st.text_input(
    "Type your question here...",
    placeholder="Ask your question...",
    key="temp_input",  # This key is used for the input field
    on_change=process_query,  # Trigger the callback when the input changes (Enter key pressed)
)

# Display chat history
st.markdown("### Chat History")
chat_container = st.container()
with chat_container:
    for entry in st.session_state.chat_history:
        sender_class = "chat-bubble-user" if entry['sender'] == 'user' else "chat-bubble-bot"
        st.markdown(
            f"""
            <div class="{sender_class}">
                {entry['message']}
                <div class="timestamp">{entry['timestamp']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
