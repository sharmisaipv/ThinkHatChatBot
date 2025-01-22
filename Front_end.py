import streamlit as st
import requests
from datetime import datetime

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

# Callback function to process the query
def process_query():
    query = st.session_state.temp_input  # Get the current input value
    if not query.strip():
        add_to_chat("⚠️ Please enter a query!", sender="bot")
        return

    add_to_chat(f"You: {query}", sender="user")
    add_to_chat("⏳ Processing query... Please wait.", sender="bot")
    
    # Send the query to the server
    with st.spinner("Sending query to the server..."):
        try:
            response = requests.post(
                "http://127.0.0.1:5000/query",  # Flask server endpoint
                json={"query": query}
            )
            if response.status_code == 200:
                data = response.json()
                bot_response = data.get("response", "No response received.")
                add_to_chat(f"🤖 Thinkhat Bot: {bot_response}", sender="bot")
            else:
                add_to_chat(f"❌ Error {response.status_code}: Unable to process the query.", sender="bot")
        except Exception as e:
            add_to_chat(f"❌ Error: {e}", sender="bot")
    
    # Clear the input field by resetting the session state value
    st.session_state.temp_input = ""  # Callback allows modification of the key safely

# Custom styles for chat messages
st.markdown(
    """
    <style>
    .chat-container {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 20px;
        height: 400px;
        overflow-y: auto;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .chat-bubble-user {
        background-color: #E8F5E9;
        color: #1B5E20;
        border-radius: 10px;
        padding: 10px 15px;
        margin: 5px;
        text-align: right;
        max-width: 75%;
        margin-left: auto;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .chat-bubble-bot {
        background-color: #E3F2FD;
        color: #0D47A1;
        border-radius: 10px;
        padding: 10px 15px;
        margin: 5px;
        text-align: left;
        max-width: 75%;
        margin-right: auto;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .timestamp {
        font-size: 10px;
        color: #999;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Chat history container
st.markdown("### Chat History")
chat_container = st.container()
with chat_container:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for entry in st.session_state.chat_history:
        message_html = f"""
        <div class="{'chat-bubble-user' if entry['sender'] == 'user' else 'chat-bubble-bot'}">
            {entry['message']}
            <div class="timestamp">{entry['timestamp']}</div>
        </div>
        """
        st.markdown(message_html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Input box for user query with callback
st.markdown("### Enter Your Query")
st.text_input(
    "Type your question here...",
    placeholder="Ask your question...",
    key="temp_input",  # This key is used for the input field
    on_change=process_query,  # Trigger the callback when the input changes (Enter key pressed)
)