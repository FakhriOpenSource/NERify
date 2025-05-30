import os
import streamlit as st
from utils import extract_text, run_ner, filter_entities
import pandas as pd

st.set_page_config(page_title="NER Ekstraktor Dokumen", layout="wide")

# Load style CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("📄NERify")

uploaded_file = st.file_uploader("Unggah file dokumen (.pdf, .docx)", type=["pdf", "docx"])

if uploaded_file is not None:
    os.makedirs("uploads", exist_ok=True)
    file_path = os.path.join("uploads", uploaded_file.name)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    text = extract_text(file_path)
    st.subheader("📖 Isi Dokumen")
    st.text_area("Teks dari dokumen:", text[:3000], height=250)

    with st.spinner("🔍 Mengekstrak entitas..."):
        df_entities = run_ner(text)

    st.subheader("📌 Filter Entitas")
    entity_types = st.multiselect("Pilih entitas yang ingin ditampilkan", 
                                  options=df_entities["Entity"].unique().tolist(), 
                                  default=df_entities["Entity"].unique().tolist())
    min_conf = st.slider("Tingkat kepercayaan minimum", 0.0, 1.0, 0.90, step=0.01)

    filtered_df = filter_entities(df_entities, entity_types, min_conf)

    st.subheader("📋 Hasil Ekstraksi")
    st.dataframe(filtered_df, use_container_width=True)

    csv = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Unduh hasil sebagai CSV",
        data=csv,
        file_name="hasil_ekstraksi_ner.csv",
        mime="text/csv"
    )
