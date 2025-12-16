import streamlit as st
import pandas as pd
import numpy as np
import re
from collections import Counter
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# =========================================================
#                      PAGE CONFIG
# =========================================================
st.set_page_config(page_title="Pencarian Obat Berdasarkan Gejala", layout="wide")

# =========================================================
#                           CSS
# =========================================================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: #F5F8F9 !important;
}

.header-title {
    text-align: center;
    font-size: 34px;
    font-weight: 600;
    margin-bottom: 30px;
    color: #1f2a37;
}

.search-box-custom textarea {
    border-radius: 40px !important;
    padding: 15px 22px !important;
    font-size: 17px !important;
    background: #8792AE !important;
}

.search-button button {
    background: #8792AE !important;
    color: #1A5F7A !important;
    border-radius: 30px !important;
    padding: 8px 35px !important;
    border: 2px solid #1A5F7A !important;
    font-size: 18px !important;
}

.result-card {
    background: #ffffff;
    border-radius: 18px;
    padding: 18px 20px;
    margin-bottom: 15px;
    border: 1px solid #D1E7F5;
    box-shadow: 0 0 15px rgba(30, 144, 255, 0.3);
}

.obat-title {
    font-size: 20px;
    font-weight: 700;
    color: #1A5F7A;
}

.score {
    font-size: 14px;
    color: #666;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
#                       HEADER
# =========================================================
st.markdown("<div class='header-title'>PENCARIAN OBAT BERDASARKAN GEJALA</div>", unsafe_allow_html=True)

# =========================================================
#                 INPUT GEJALA
# =========================================================
gejala = st.text_area("", height=60, label_visibility="collapsed")

jumlah_kata = len(gejala.strip().split())

if jumlah_kata < 2 and gejala.strip() != "":
    st.markdown("""
    <style>
    textarea {
        color: red !important;
        border: 2px solid red !important;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown("<div style='color:red;text-align:center'>Gejala minimal 2 kata</div>", unsafe_allow_html=True)

run = st.button("Cari", disabled=(jumlah_kata < 2))

# =========================================================
#                     LOAD DATA
# =========================================================
df = pd.read_csv("data_obat.csv")
gt_df = pd.read_excel("data_query.xlsx")

stop_factory = StopWordRemoverFactory()
stopwords = set(stop_factory.get_stop_words())
tanda_baca = ['.', ',', '-', ':', ';', '/', '(', ')']

# =========================================================
#                     PREPROCESSING
# =========================================================
def cleaning(text):
    text = re.sub(r"[0-9]", "", text.lower())
    for ch in tanda_baca:
        text = text.replace(ch, "")
    return text

def tokenize(t): return t.split()
def remove_stopwords(tok): return [w for w in tok if w not in stopwords]
def ngram(tokens, n): return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

df["clean"] = df["Kegunaan obat"].astype(str).apply(cleaning).apply(tokenize).apply(remove_stopwords)
df["unigram"] = df["clean"].apply(lambda x: ngram(x,1))
df["bigram"] = df["clean"].apply(lambda x: ngram(x,2))

nama_obat_list = df["Nama obat"].tolist()

# =========================================================
#                     TF-IDF
# =========================================================
def build_tfidf(list_ngram):
    vocab = sorted(set(w for doc in list_ngram for w in doc))
    base = pd.DataFrame({"kata": vocab})

    for nama, doc in zip(nama_obat_list, list_ngram):
        cnt = Counter(doc)
        N = len(doc)
        base[nama] = base["kata"].apply(lambda k: cnt.get(k,0)/N if N>0 else 0)

    base["df"] = base.apply(lambda r: sum(r[n] > 0 for n in nama_obat_list), axis=1)
    base["IDF"] = base["df"].apply(lambda d: np.log(len(nama_obat_list)/d) if d>0 else 0)

    for n in nama_obat_list:
        base[n] *= base["IDF"]

    return base

tfidf_uni = build_tfidf(df["unigram"])
tfidf_bi  = build_tfidf(df["bigram"])

# =========================================================
#                     QUERY VECTOR
# =========================================================
def proses_gejala(text):
    t = remove_stopwords(tokenize(cleaning(text)))

    g1 = Counter(ngram(t,1))
    v1 = np.array([g1.get(k,0) for k in tfidf_uni["kata"]]) * tfidf_uni["IDF"].values

    g2 = Counter(ngram(t,2))
    v2 = np.array([g2.get(k,0) for k in tfidf_bi["kata"]]) * tfidf_bi["IDF"].values

    return v1, v2

# =========================================================
#                     COSINE SEARCH
# =========================================================
def cari_obat(text, mode="unigram"):
    v1, v2 = proses_gejala(text)
    v = v1 if mode == "unigram" else v2
    tfidf = tfidf_uni if mode == "unigram" else tfidf_bi

    hasil = []
    nv = np.linalg.norm(v)

    for nama in nama_obat_list:
        d = tfidf[nama].values
        nd = np.linalg.norm(d)
        skor = np.dot(v,d)/(nv*nd) if nv>0 and nd>0 else 0
        ket = df[df["Nama obat"] == nama]["Kegunaan obat"].values[0]
        hasil.append([nama, skor, ket])

    return pd.DataFrame(hasil, columns=["Nama Obat","Skor","Keterangan"]).sort_values("Skor", ascending=False)

# =========================================================
#               GROUND TRUTH & PRECISION
# =========================================================
def get_ground_truth(q):
    row = gt_df[gt_df["query"].str.lower() == q.lower()]
    if row.empty:
        return []
    return [x.strip() for x in row.iloc[0]["daftar_obat"].split(",")]

def precision_at_3(df_hasil, gt):
    top3 = df_hasil.head(3)["Nama Obat"].tolist()
    relevan = [o for o in top3 if o in gt]
    return len(relevan)/3

# =========================================================
#                     OUTPUT
# =========================================================
if run and gejala.strip() != "":

    col1, col2 = st.columns(2)
    gt = get_ground_truth(gejala)

    with col1:
        st.subheader("UNIGRAM")
        h = cari_obat(gejala,"unigram")
        if h["Skor"].max() == 0:
            st.info("Tidak ada kata yang cocok")
        else:
            for _,r in h.head(3).iterrows():
                st.markdown(f"<div class='result-card'><div class='obat-title'>{r['Nama Obat']}</div><div class='score'>Skor: {r['Skor']:.4f}</div><p>{r['Keterangan']}</p></div>", unsafe_allow_html=True)
            if gt:
                st.markdown(f"<b>Precision@3:</b> {precision_at_3(h,gt):.2f}", unsafe_allow_html=True)

    with col2:
        st.subheader("BIGRAM")
        h = cari_obat(gejala,"bigram")
        if h["Skor"].max() == 0:
            st.info("Tidak ada kata yang cocok")
        else:
            for _,r in h.head(3).iterrows():
                st.markdown(f"<div class='result-card'><div class='obat-title'>{r['Nama Obat']}</div><div class='score'>Skor: {r['Skor']:.4f}</div><p>{r['Keterangan']}</p></div>", unsafe_allow_html=True)
            if gt:
                st.markdown(f"<b>Precision@3:</b> {precision_at_3(h,gt):.2f}", unsafe_allow_html=True)

