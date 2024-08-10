import os
import pandas as pd
import torch
import numpy as np
import faiss
import pickle
from transformers import AutoModel, AutoTokenizer
from torch.nn.functional import normalize
from groq import Groq
import warnings


warnings.filterwarnings("ignore")


os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
    query = f'Represent this sentence for searching relevant passages: {query}'
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
        model="mixtral-8x7b-32768",
    )
    return response.choices[0].message.content

def main():
    
    print("Welcome to the Xtracap Fintech Policy Document Chatbot!")
    print("Type 'stop' to end the session.\n")

    
    chunk_embeddings, index = load_embeddings_and_index(embeddings_file, index_file)
    client = Groq(api_key="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")

    while True:
        
        query = input("Enter your query: ")

        if query.strip().lower() == "stop":
            print("Ending the session. Goodbye!")
            break
        elif query.strip() == "":
            print("Please enter a valid query.")
        else:
            print("Retrieving and processing the answer...")
            relevant_chunks = retrieve_relevant_chunks(query, model_id, index, df, cache_dir=cache_dir)
            response = generate_response(query, relevant_chunks, client)
            
            print("Answer:")
            print(response)
            print("\n")  

if __name__ == "__main__":
    main()
