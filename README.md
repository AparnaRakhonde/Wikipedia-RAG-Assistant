# Wikipedia RAG Assistant

The **Wikipedia RAG Assistant** is a simple Retrieval-Augmented Generation (RAG) project that allows you to query Wikipedia articles and get precise answers using Natural Language Processing (NLP).

---

## 🚀 Features
- Fetch content directly from **Wikipedia**
- Use **FAISS** for similarity search
- Apply **Sentence Transformers** for embeddings
- Use a **Hugging Face QA model** to extract answers
- Interactive interface with **Streamlit**

---

## 📂 Project Structure
```
Wikipedia-RAG-Assistant/
│
├── app.py             # Streamlit app
├── rag_pipeline.py    # Core logic for retrieval and QA
├── requirements.txt   # Dependencies
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

1. Clone the repository:
```bash
git clone https://github.com/AparnaRakhonde/Wikipedia-RAG-Assistant.git
cd Wikipedia-RAG-Assistant
```

2. Create a virtual environment:
```bash
python -m venv venv
# Activate on Windows
venv\Scripts\activate
# Activate on Linux / macOS
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the app with:
```bash
streamlit run app.py
```

---

## 🔍 Example Q&A Interaction

**Topic:** Machine Learning  
**Question:** What is supervised learning?  
**Answer:** A type of machine learning where models are trained on labeled data.  

> You can try any topic available on Wikipedia and ask related questions. The assistant will retrieve relevant chunks and provide an answer based on the content.

---

## ⚙️ How It Works

1. **Fetch Data** → User enters a topic, and the app retrieves the related Wikipedia article.  
2. **Chunking & Embeddings** → The article is split into smaller chunks and converted into embeddings using a Sentence Transformer model.  
3. **Similarity Search** → FAISS is used to find the most relevant chunks for the user’s query.  
4. **Answer Extraction** → A Hugging Face QA model (e.g., `deepset/roberta-base-squad2`) generates the final answer from the retrieved text.  

---

## 🔮 Future Improvements
- Add support for multiple QA models
- Improve UI/UX in Streamlit
- Enable offline Wikipedia cache
- Deploy on Hugging Face Spaces or Streamlit Cloud

---

👩‍💻 Author: **Aparna Ganesh Rakhonde**
