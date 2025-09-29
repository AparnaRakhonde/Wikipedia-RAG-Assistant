import wikipedia
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from sentence_transformers import SentenceTransformer

# Models
embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
qa_model_name = "deepset/roberta-base-squad2"
qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)
qa_pipeline = pipeline('question-answering', model=qa_model, tokenizer=qa_tokenizer)

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')


def get_wikipedia_content(topic):
    try:
        page = wikipedia.page(topic)
        return page.content
    except (wikipedia.exceptions.PageError, wikipedia.exceptions.DisambiguationError):
        return None
    

def split_text(text, chunk_size=1000, chunk_overlap=30):
    tokens = tokenizer.tokenize(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append(tokenizer.convert_tokens_to_string(tokens[start:end]))
        if end == len(tokens):
            break
        start = end - chunk_overlap
    return chunks

def build_faiss_index(chunks):
    embeddings = embedding_model.encode(chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index, embeddings


def retrieve_chunks(query, chunks,embeddings, index, k=3):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)
    return [chunks[i] for i in indices[0]]


def answer_question(question, context):
    result = qa_pipeline(question=question, context=context)
    return result['answer']