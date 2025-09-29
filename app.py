import streamlit as st
from rag_pipeline import (
    get_wikipedia_content,
    split_text,
    build_faiss_index,
    retrieve_chunks,
    answer_question
)

st.set_page_config(page_title="Wikipedia RAG Assistant", layout="centered")
st.title("📚 Wikipedia RAG Assistant")

# --- Step 1 : Topic input ---
topic = st.text_input("🔎 Enter a topic to search on wikipedia : ")

if st.button("Retrive and learn") and topic:
    with st.spinner("Fetching content from wikipedia..."):
        document = get_wikipedia_content(topic)

    if not document:
        st.error("❌ Could not retrieve content. Try a different topic.")
    else:
        st.success("✅ Topic retrieved successfully!")

        chunks = split_text(document)
        index, embeddings = build_faiss_index(chunks)

        # Store in session state
        st.session_state.chunks = chunks
        st.session_state.index = index
        st.session_state.embeddings = embeddings
        


# --- Step 2: Question Answering ---

if "chunks" in st.session_state:
    question = st.text_input("💬 Ask a question about the topic: ")

    if st.button("Get Answer") and question:
        with st.spinner("🔎 Retrieving relevant chunks and generating answer..."):
            chunks = st.session_state.chunks
            index = st.session_state.index
            embeddings = st.session_state.embeddings

            retrieved = retrieve_chunks(question, chunks, embeddings, index)

            if not retrieved:
                st.warning("⚠️ No relevant chunks found. Try rephrasing your question.")
            else:
                context = " ".join(retrieved)
                answer = answer_question(question, context)

                st.subheader("✅ Answer : ")
                st.success(answer)

                with st.expander("📚 Retrieved Chunks"):
                    for c in retrieved:
                        st.write("- "+ c)
                        
