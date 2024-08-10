import os
import pandas as pd
import torch
import numpy as np
import faiss
import pickle
from transformers import AutoModel, AutoTokenizer
from torch.nn.functional import normalize
from groq import Groq
import streamlit as st
from PIL import Image

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

folder_path = r'/Users/akshatb/documents/ragfin/foldeer'
model_id = "Snowflake/snowflake-arctic-embed-m-v1.5"
device = "cuda" if torch.cuda.is_available() else "cpu"
embeddings_file = 'embeddings.pkl'
index_file = 'faiss_index.bin'
cache_dir = r"/Users/akshatb/documents/ragfin/cache"

df = pd.read_csv(r'/Users/akshatb/documents/ragfin/new.csv')

def load_embeddings_and_index(embeddings_file, index_file):
    with open(embeddings_file, 'rb') as f:
        chunk_embeddings = pickle.load(f)
    index = faiss.read_index(index_file)
    return chunk_embeddings, index

def retrieve_relevant_chunks(query, model_id, index, df, cache_dir, k=5):
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    model = AutoModel.from_pretrained(model_id, add_pooling_layer=False, cache_dir=cache_dir)
    model.eval()

    query_tokens = tokenizer([query], padding=True, truncation=True, return_tensors='pt', max_length=512)
    with torch.inference_mode():
        query_embedding = model(**query_tokens)[0][:, 0]
    query_embedding = normalize(query_embedding).numpy().astype(np.float32)

    D, I = index.search(query_embedding, k)
    relevant_chunks = [df.iloc[i] for i in I[0]]
    return relevant_chunks

def generate_response(query, relevant_chunks, client):
    context = "\n".join([f"From {chunk['filename']}: {chunk['chunk']}" for chunk in relevant_chunks])
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": (
                    f"Based on the following excerpts from policy documents:\n\n"
                    f"{context}\n\n"
                    f"Please provide a precise and accurate response to the query below. The answer should be directly derived from the provided context, "
                    f"and be limited to 100 words. Ensure the response reflects the formal and authoritative nature of the policy documents, "
                    f"and avoid any creative interpretation or irrelevant information:\n\n"
                    f"Query: {query}\n"
                )
            }
        ],
        model="llama-3.1-8b-instant",
    )
    return response.choices[0].message.content

def main():
    st.set_page_config(page_title="Xtracap Fintech Document Query", page_icon=":book:", layout="wide")

    # Custom CSS for styling
    st.markdown(
        """
        <style>
        .main {
            background-color: #f5f5f5;
            color: #333;
        }
        .header {
            text-align: center;
            color: #0077b5; /* Xtracap Fintech blue */
        }
        .stButton>button {
            background-color: #0077b5;
            color: white;
        }
        .stTextInput>div>input {
            background-color: #ffffff;
            color: #333;
        }
        .stMarkdown h1 {
            color: #0077b5;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Display company logo and heading
    logo_image = Image.open(r'C:\Users\akshatb\Worj\rag\rag\Picture1.jpg')  # Update this path to your image location
    st.image(logo_image, width=150)  # Logo size adjusted

    st.markdown("<h1 class='header'>Xtracap Fintech Policy Document Chatbot</h1>", unsafe_allow_html=True)

    # App description
    st.write("Welcome to the Xtracap Fintech Policy Document Chatbot. Enter your query below to get answers based on our policy documents.")

    # User input for query
    query = st.text_input("Your Query:", placeholder="Type your question here...")

    # Button to get the answer
    if st.button("Get Answer"):
        if query.strip() == "":
            st.warning("Please enter a query before clicking 'Get Answer'.", icon="â ï¸")
        else:
            chunk_embeddings, index = load_embeddings_and_index(embeddings_file, index_file)
            client = Groq(api_key="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")
            
            with st.spinner("Retrieving and processing the answer..."):
                relevant_chunks = retrieve_relevant_chunks(query, model_id, index, df, cache_dir=cache_dir)
                response = generate_response(query, relevant_chunks, client)
            
            st.write("**Answer:**")
            st.write(response)

if __name__ == "__main__":
    main()
