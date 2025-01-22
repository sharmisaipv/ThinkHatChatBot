# ThinkHat Chatbot 🤖

ThinkHat Chatbot is an intelligent assistant built to answer user queries efficiently. This project leverages advanced machine learning models like `SentenceTransformer`, `FAISS`, and `Flan-T5` to provide accurate and meaningful responses to user questions.

---

## Features
- **Interactive Chat Interface**: A Streamlit-based front end for seamless interaction.
- **Backend Intelligence**: A Flask server managing queries and responses using FAISS for semantic search.
- **Summarization and Generation**: Summarizes large content and generates meaningful insights using T5 and BART models.
- **Modular Design**: Clean separation of backend and frontend code.

---

## Prerequisites
Before running the project, ensure you have the following installed:
- Python 3.8+
- Virtual environment (recommended)
- Git

---

## Installation

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/sharmisaipv/ThinkHatChatBot.git
   cd ThinkHatChatBot

2. **Set Up Virtual Environment**
python -m venv thinkhat_env
source thinkhat_env/bin/activate   # For Linux/Mac
thinkhat_env\Scripts\activate     # For Windows

3. **Install Dependencies**
pip install -r requirements.txt
