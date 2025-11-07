import gc
import torch
from torch.quantization import quantize_dynamic
import streamlit as st
import numpy as np
import pandas as pd
import faiss
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipModel
from sentence_transformers import SentenceTransformer
from PIL import Image
import requests
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os

# --- 1. Quantized CLIP & lazy‑loaded SBERT ---
torch.backends.quantized.engine = "qnnpack"



load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

query_slicing_system_msg = {
    "role": "system",
    "content": """
    You are a smart query parser for a multi-modal product search system.
    Your job is to split any incoming query into two precise outputs: (1) star/rating/price filters and (2) product search terms.

    Format your response as:
    output_1: Extract only information related to star ratings, customer reviews, or price (e.g., "stars above 4", "price under 50"). Summarize clearly and concisely.
    output_2: Extract the remaining product-related details (such as item type, material, finish, color, style, etc.) that are helpful for visual or embedding-based similarity search.
        Remove any duplicates or redundant phrases, and return the product name first with clean, lowercase, comma-separated keywords suitable for retrieval.
    """
}

recommendation_system_msg = {
    "role": "system",
    "content": """
    You are an intelligent, frinedly, and energentic senior shopping assistant. A user is trying to find the right product based on their preferences.
    You will get the user query, user preferences, and the products retrieved by existing system based on visual and textual similarity to the user's preferences.
    Your goal is to provide the best recommendation, referencing following instructions.

    Instructions:
    1. Recommend 1-3 products from the retrieved products that best match the user's preferences. Prioritize products that match the user’s specific product description (e.g., design, style, material, target audience) in the query_text first.
    Only consider price and rating *after* verifying the product is relevant. Only recommend products provided from the "Retrieved Products" input.
    2. Only mention recommended products in your response. Do not mention products that were not selected.
    3. For each recommended product, provide:
      - Product Name
      - A short summary
      - Price
      - Rating
      - Pros and cons (if relevant)
      - Image URL (include as a standalone link to help user preview visually)
      - Product URL if available (optional, include only if exists)
    4. Be clear, friendly, energentic. Format recommendations in an easy-to-read way.
    5. Do not mention product numbers or internal labels (e.g., "Product 3").
    6. Provide a short final recommendation summary.

    Now generate your recommendation:
"""
}




@st.cache_resource(show_spinner=False)
def load_clip():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model.to("cpu")
    from torch.quantization import quantize_dynamic
    # this will now use QNNPACK under the hood
    model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

# we'll only call this if the user picks Clip‑STrans
@st.cache_resource(show_spinner=False)
def load_strans():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def load_blip():
    # Load and eval
    model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
    model.eval()
    # Quantize to qint8 (CPU only)
    model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    # Load tokenizer/processor
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    return model, processor

# --- 2. Memory‑map FAISS + csv data + downcast dtypes ---
@st.cache_data(show_spinner=False)
def load_data_and_indices(model_choice):
    # ——— Load only needed CSV columns, memory‑map, and downcast dtypes ———
    df = pd.read_csv(
        "data/Amazon2023DS_partial_NLP.csv",
        usecols=["image", "title", "store", "price", "average_rating"],
        dtype={
            "store": "category",
            "price": "float32",
            "average_rating": "float32",
        },
        memory_map=True,
        low_memory=False,
    )

    # ——— Memory‑map FAISS indexes ———
    io_flag = faiss.IO_FLAG_MMAP
    if model_choice == "Clip-Clip":
        text_idx = faiss.read_index("data/clip_text_vector_index_w_NLP_2.faiss", io_flag)
        img_idx  = faiss.read_index("data/clip_image_vector_index.faiss", io_flag)
    elif model_choice == "Clip-STrans":
        text_idx = faiss.read_index("data/SBERT_text_vector_index_w_NLP.faiss", io_flag)
        img_idx  = faiss.read_index("data/clip_image_vector_index.faiss", io_flag)
    elif model_choice == "Clip-Blip":
        text_idx = faiss.read_index("data/BLIP_text_vector_index_w_NLP.faiss", io_flag)
        img_idx  = faiss.read_index("data/clip_image_vector_index.faiss", io_flag)
    elif model_choice == "Blip-Text":
        text_idx = faiss.read_index("data/faiss_amazon_index.faiss", io_flag)
        img_idx  = faiss.read_index("data/clip_image_vector_index.faiss", io_flag)
    else:  # Blip‑Blip fallback
        text_idx = faiss.read_index("data/clip_text_vector_index_w_NLP_2.faiss", io_flag)
        img_idx  = faiss.read_index("data/clip_image_vector_index.faiss", io_flag)

    return df, text_idx, img_idx


def get_clip_text_embedding(text: str, clip_model, clip_processor) -> np.ndarray:
    inputs = clip_processor(text=[text], return_tensors="pt")
    with torch.no_grad():
        feats = clip_model.get_text_features(**inputs)
    arr = feats.squeeze().cpu().detach().numpy().astype("float32").reshape(1, -1)
    return arr

def get_sbert_text_embedding(text: str, strans_model) -> np.ndarray:
    emb = strans_model.encode(text, normalize_embeddings=True)
    return emb.reshape(1, -1).astype("float32")

def get_blip_text_embedding(text:str, blip_model, blip_processor):
    """Generate BLIP text embedding for a single text query."""
    inputs = blip_processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        output = blip_model.text_model(**inputs).last_hidden_state[:, 0, :]  # CLS token
    embedding = output.cpu().numpy()
    embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)  # normalize
    return embedding.astype("float32")

def retrieve_similar_products(
    query: str,
    k: int = 5,
    alpha: float = 0.5,
    model_choice: str = "Clip-STrans",
    multiplier: int = 20,  
) -> pd.DataFrame:
    df, text_idx, img_idx = load_data_and_indices(model_choice)
    clip_model, clip_processor = load_clip()

    # lazy‑load SBERT only if needed
    strans_model = load_strans() if model_choice == "Clip-STrans" or model_choice == "Blip-Text" else None
    blip_model, blip_processor = load_blip() if model_choice == "Clip-Blip" else (None, None)

    # embed + search
    if model_choice == "Clip-Clip":
        emb_txt = get_clip_text_embedding(query, clip_model, clip_processor)
        emb_img = get_clip_text_embedding(query, clip_model, clip_processor)
        faiss.normalize_L2(emb_txt)
        faiss.normalize_L2(emb_img)
        td, ti = text_idx.search(emb_txt, k * multiplier)
        id_, ii = img_idx.search(emb_img, k * multiplier)

    elif model_choice == "Clip-STrans":
        emb_img = get_clip_text_embedding(query, clip_model, clip_processor)
        emb_txt = get_sbert_text_embedding(query, strans_model)
        faiss.normalize_L2(emb_txt)
        faiss.normalize_L2(emb_img)
        td, ti = text_idx.search(emb_txt, k * multiplier)
        id_, ii = img_idx.search(emb_img, k * multiplier)
    elif model_choice == "Clip-Blip":
        emb_img = get_clip_text_embedding(query, clip_model, clip_processor)
        emb_txt = get_blip_text_embedding(query, blip_model, blip_processor)
        faiss.normalize_L2(emb_txt)
        faiss.normalize_L2(emb_img)
        td, ti = text_idx.search(emb_txt, k * multiplier)
        id_, ii = img_idx.search(emb_img, k * multiplier)
    elif model_choice == "Blip-Text":
        emb_txt = get_sbert_text_embedding(query, strans_model)
        faiss.normalize_L2(emb_txt)
        td, ti = text_idx.search(emb_txt, k)
        results = df.iloc[ii].copy()
        for name in ('emb_txt', 'td', 'ti'):
            if name in locals():
                del locals()[name]
        gc.collect()
        return results
    else:  # Blip‑Blip fallback to Clip‑Clip
        emb_txt = get_clip_text_embedding(query, clip_model, clip_processor)
        emb_img = get_clip_text_embedding(query, clip_model, clip_processor)
        faiss.normalize_L2(emb_txt)
        faiss.normalize_L2(emb_img)
        td, ti = text_idx.search(emb_txt, k * multiplier)
        id_, ii = img_idx.search(emb_img, k * multiplier)

    # aggregate scores
    candidates = list(set(ti[0]) | set(ii[0]))
    similarity_scores = {}
    for idx in candidates:
      product_text_emb = text_idx.reconstruct(int(idx)).reshape(1, -1)
      product_image_emb = img_idx.reconstruct(int(idx)).reshape(1, -1)

      text_sim = cosine_similarity(emb_txt, product_text_emb)[0][0]
      image_sim = cosine_similarity(emb_img, product_image_emb)[0][0]

      similarity_scores[idx] = alpha * text_sim + (1 - alpha) * image_sim

    # Step 5: Rank Results
    sorted_indices = sorted(similarity_scores, key=similarity_scores.get, reverse=True)[:k]

    # Retrieve final ranked products
    results = df.iloc[sorted_indices].copy()

    # --- cleanup to free memory between runs ---
    for name in ('emb_txt', 'emb_img', 'td', 'ti', 'id_', 'ii'):
        if name in locals():
            del locals()[name]
    gc.collect()

    return results

@st.cache_data(show_spinner=False, max_entries=50)
def slice_query(query_text: str) -> tuple[str, str]:
    messages = [
        query_slicing_system_msg,
        {"role": "user", "content": f'Query:\n"{query_text}"'}
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    content = response.choices[0].message.content
    lines = content.strip().split('\n')
    output_1 = next((line.replace("output_1:", "").strip() for line in lines if "output_1" in line.lower()), "")
    output_2 = next((line.replace("output_2:", "").strip() for line in lines if "output_2" in line.lower()), "")

    # print(f"Output 1 = {output_1}, Output 2 = {output_2}")

    return output_1, output_2
# And you can also cache the final RAG response if you like:
@st.cache_data(show_spinner=False, max_entries=50)
def generate_rag_response(query: str, k: int = 5, alpha: float = 0.5, model_choice: str = "Clip-STrans") -> str:
    output_1, output_2 = slice_query(query)
    retrieved_docs = retrieve_similar_products(output_2, k=k, alpha=alpha, model_choice=model_choice)

    context = "\n\n".join([
        f"Title: {row['title']}\n"
        f"Description: {row.get('description', '')}\n"
        f"Store: {row.get('store', '')}\n"
        f"Features: {row.get('features', '')}\n"
        f"Details: {row.get('details', '')}\n"
        f"Price: {row.get('price', 'N/A')}\n"
        f"Rating: {row.get('average_rating', 'N/A')}\n"
        f"Image: {row['image']}"
        for _, row in retrieved_docs.iterrows()
    ])

    # print("Retrieved products \n", context)

    messages = [
        recommendation_system_msg,
        {"role": "user", "content": f'''
        Query: "{query}"

        User preferences (price/rating filters):
        {output_1}

        Retrieved Products:
        {context}
        '''}
        ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    return response.choices[0].message.content

# --- Streamlit UI remains largely unchanged ---
st.set_page_config(page_title="Zuzu", layout="centered")
st.title("Hi, I'm Zuzu .‿. - Your personal shopping assistant!")
st.subheader("What can I help you with today?")

col1_, col2_ = st.columns([0.8, 0.2])
with col1_:
    query = st.text_input("Type your question here…", key="input")
with col2_:
    mode_choice = st.radio("Tech Mode:", ["ON", "OFF"], index = 1)


if mode_choice == "ON":
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        alpha = st.slider("Alpha:", 0.0, 1.0, 0.5)
    with col2:
        model_choice = st.radio("Model:", ["Clip-Clip", "Clip-STrans", "Clip-Blip", "Blip-Text"])
else:
    alpha = 0.15
    model_choice = "Clip-STrans"

if st.button("Enter") and query:
    rag_resp = generate_rag_response(query, k=10, alpha=alpha, model_choice=model_choice)
    st.markdown("### You asked:")
    st.write(query)
    st.markdown("### 💡 Model Response:")
    st.markdown(rag_resp)
else:
    st.write("Awaiting your question…")
