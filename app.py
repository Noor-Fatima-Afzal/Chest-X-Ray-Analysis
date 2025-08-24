import os
import ast
import io
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import torch
import faiss

from transformers import CLIPProcessor, CLIPModel
from openai import OpenAI

# ----------------------------
# Config
# ----------------------------
CSV_PATH = "/content/mimic_cxr/mimic_cxr_data.csv"
TOP_K_DEFAULT = 3
N_INDEX_ROWS = 100  
MODEL_NAME_DEFAULT = "llama-3.1-8b-instant" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# Streamlit page config
# ----------------------------
st.set_page_config(page_title="Multimodal RAG for CXR (Groq)", layout="wide")

st.title("🩻 Multimodal RAG (Image + Text) for Chest X-rays")
st.caption("Retrieves similar MIMIC-CXR reports (first 100 rows) and asks a Groq LLM to draft Findings & Impression.")

# ----------------------------
# Secrets / Env
# ----------------------------
# Prefer st.secrets if available, otherwise fall back to env var
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
if not GROQ_API_KEY:
    st.warning("⚠️ Please set GROQ_API_KEY in Streamlit secrets or environment variables to enable LLM calls.")

# ----------------------------
# Load models (cached)
# ----------------------------
@st.cache_resource(show_spinner=True)
def load_clip():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    return model, processor

clip_model, clip_processor = load_clip()

# ----------------------------
# Helper functions
# ----------------------------
def str_to_bytes(x):
    """Convert "b'...'" strings back to bytes using ast.literal_eval."""
    if isinstance(x, bytes):
        return x
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            # Fallback: try to interpret as raw bytes string without the b'' wrapper
            return x.encode("latin-1")
    return x

@torch.no_grad()
def embed_image_text_bytes(image_bytes: bytes, text: str) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Image
    image_inputs = clip_processor(images=image, return_tensors="pt").to(DEVICE)
    img_feat = clip_model.get_image_features(**image_inputs)

    # Text (truncate to CLIP's 77 token limit)
    text_inputs = clip_processor(
        text=[text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77
    ).to(DEVICE)
    txt_feat = clip_model.get_text_features(**text_inputs)

    # Normalize & concat
    img_feat = torch.nn.functional.normalize(img_feat, dim=-1)
    txt_feat = torch.nn.functional.normalize(txt_feat, dim=-1)
    return torch.cat((img_feat, txt_feat), dim=-1).squeeze().cpu().numpy()

@torch.no_grad()
def embed_user_query(image_pil: Image.Image, text: str) -> np.ndarray:
    image_inputs = clip_processor(images=image_pil, return_tensors="pt").to(DEVICE)
    img_feat = clip_model.get_image_features(**image_inputs)

    text_inputs = clip_processor(
        text=[text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77
    ).to(DEVICE)
    txt_feat = clip_model.get_text_features(**text_inputs)

    img_feat = torch.nn.functional.normalize(img_feat, dim=-1)
    txt_feat = torch.nn.functional.normalize(txt_feat, dim=-1)
    return torch.cat((img_feat, txt_feat), dim=-1).squeeze().cpu().numpy().reshape(1, -1)

def build_faiss(emb_matrix: np.ndarray) -> faiss.IndexFlatL2:
    dim = emb_matrix.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(emb_matrix)
    return index

def groq_llm_answer(client: OpenAI, model_name: str, user_query: str, retrieved):
    context = "\n\n".join([
        f"[Case {i+1}]\nFindings: {r['findings']}\nImpression: {r['impression']}"
        for i, r in enumerate(retrieved)
    ])

    prompt = f"""
You are a radiology assistant. Based on the user's uploaded image and their query, and using similar past radiology cases (provided below), produce the following:

## Findings
Provide a detailed textual analysis of the chest X-ray.

## Impression
A short, clinically-focused diagnostic summary.

User query: "{user_query}"

Relevant past cases:
{context}
"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# ----------------------------
# Load data + build index (cached)
# ----------------------------
@st.cache_resource(show_spinner=True)
def prepare_index(csv_path: str, n_rows: int):
    st.info("Loading CSV and preparing FAISS index (first %d rows)..." % n_rows)
    df = pd.read_csv(csv_path)

    # Make sure required columns exist
    expected_cols = {"image", "findings", "impression"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing expected columns: {missing}")

    df["image"] = df["image"].apply(str_to_bytes)

    df_sample = df.head(n_rows).reset_index(drop=True)
    embeddings = []
    for i in range(len(df_sample)):
        emb = embed_image_text_bytes(
            df_sample.loc[i, "image"],
            df_sample.loc[i, "findings"]
        )
        embeddings.append(emb)

    emb_matrix = np.stack(embeddings).astype("float32")
    index = build_faiss(emb_matrix)
    metadata = df_sample[["findings", "impression"]].to_dict(orient="records")

    return index, metadata

try:
    index, metadata = prepare_index(CSV_PATH, N_INDEX_ROWS)
    st.success(f"Index ready. Using top {N_INDEX_ROWS} rows.")
except Exception as e:
    st.error(f"Failed to load/build index: {e}")
    st.stop()

# ----------------------------
# Sidebar controls
# ----------------------------
with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Top-K retrieved reports", 1, 10, TOP_K_DEFAULT, 1)
    model_name = st.text_input("Groq model name", MODEL_NAME_DEFAULT)
    if not GROQ_API_KEY:
        st.info("Enter GROQ_API_KEY in the box below if it's not set in secrets / env.")
    manual_api_key = st.text_input("GROQ_API_KEY (optional override)", value="", type="password")
    if manual_api_key:
        GROQ_API_KEY = manual_api_key

# ----------------------------
# Main UI
# ----------------------------
uploaded_image = st.file_uploader("Upload a chest X-ray (JPEG/PNG)", type=["jpg", "jpeg", "png"])
user_question = st.text_area("Ask your question (e.g., 'What abnormalities do you see?')", "")

col1, col2 = st.columns(2)

with col1:
    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded image", use_column_width=True)

with col2:
    st.write("")

run_button = st.button("Run Multimodal RAG")

if run_button:
    if uploaded_image is None:
        st.error("Please upload an image first.")
        st.stop()
    if not user_question.strip():
        st.error("Please enter a question.")
        st.stop()
    if not GROQ_API_KEY:
        st.error("No GROQ_API_KEY provided. Set it in secrets or the sidebar.")
        st.stop()

    try:
        # Embed user image + text
        image_pil = Image.open(uploaded_image).convert("RGB")
        query_emb = embed_user_query(image_pil, user_question)

        # Search FAISS
        D, I = index.search(query_emb, top_k)
        retrieved = [metadata[i] for i in I[0]]

        # Show retrieved context
        with st.expander("🔎 Retrieved similar cases"):
            for i, r in enumerate(retrieved, 1):
                st.markdown(f"**Case {i}**")
                st.markdown(f"**Findings:** {r['findings']}")
                st.markdown(f"**Impression:** {r['impression']}")
                st.markdown("---")

        # Ask Groq LLM
        client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY)
        with st.spinner("Querying Groq LLM..."):
            answer = groq_llm_answer(client, model_name, user_question, retrieved)

        st.subheader("🧠 LLM Response")
        st.markdown(answer)

    except Exception as e:
        st.error(f"Error while running the pipeline: {e}")
