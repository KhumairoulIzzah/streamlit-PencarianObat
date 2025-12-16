import streamlit as st
import pandas as pd
import numpy as np
import re
from collections import Counter
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Pencarian Obat Berdasarkan Gejala",
    layout="wide"
)

# ===============================
# CSS
# ===============================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: #F5F8F9;
}

.header-title {
    text-align: center;
    font-size: 34px;
    font-weight: 600;
    margin-bottom: 30px;
}

textarea {
    border-radius: 20px !important;
    font-size: 16px !important;
}

.result-card {
    background: white;
    border-radius: 18px;
    padding: 18px;
    margin-bottom: 15px;
    box-shadow: 0 0 15px rgba(30,144,255,0.25);
}

.obat-title {
    font-size: 20px;
    font-weight: 700;
    color: #1A5F7A;
}

.score {
    font-size: 14px;
    color: #555;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# HEADER
# ===============================
st.markdown(
    "<div class='header-title'>PENCARIAN OBAT BERDASARKAN GEJALA</div>",
    unsafe_allow_html=True
)

# ===============================
# INPUT GEJALA
# ===============================
gejala = st.text_area("", height=70, label_visibility="collapsed")

jumlah_kata = len(gejala.strip().split())

st.markdown(
    "<div style='font-size:13px; color:#666;'>maksimal 100 kata</div>",
    unsafe_allow_html=True
)

# validasi warna merah
if (jumlah_kata < 2 and gejala.strip() != "") or jumlah_kata > 100:
    st.markdown("""
    <style>
    textarea {
        color: red !important;
        border: 2px solid red !important;
    }
    </style>
    """, unsafe_allow_html=True)

if jumlah_kata < 2 and gejala.strip() != "":
    st.markdown("<div style='color:red'>Minimal 2 kata</div>", unsafe_allow_html=True)

if jumlah_kata > 100:
    st.markdown("<div style='color:red'>Maksimal 100 kata</div>", unsafe_allow_html=True)

run = st.button(
    "Cari",
    disabled=(jumlah_kata < 2 or jumlah_kata > 100)
)

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv("data_obat.csv")
gt_df = pd.read_excel("data_query.xlsx")

stop_factory = StopWordRemoverFactory()
stopwords = set(stop_factory.get_stop_words())
tanda_baca = ['.', ',', '-', ':', ';', '/', '(', ')']

# ===============================
# PREPROCESSING
# ===============================
def cleaning(text):
    text = text.lower()
    text = re.sub(r"[0-9]", "", text)
    for t in tanda_baca:
        text = text.replace(t, "")
    return text

def tokenize(text):
    return text.split()

def remove_stopwords(tokens):
    return [t for t in tokens if t not in stopwords]

def ngram(tokens, n):
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

df["clean"] = (
    df["Kegunaan obat"]
    .astype(str)
    .apply(cleaning)
    .apply(tokenize)
    .apply(remove_stopwords)
)

df["unigram"] = df["clean"].apply(lambda x: ngram(x, 1))
df["bigram"] = df["clean"].apply(lambda x: ngram(x, 2))

nama_obat_list = df["Nama obat"].tolist()

# ===============================
# TF-IDF
# ===============================
def build_tfidf(data_ngram):
    vocab = sorted(set(w for doc in data_ngram for w in doc))
    base = pd.DataFrame({"kata": vocab})

    for nama, doc in zip(nama_obat_list, data_ngram):
        cnt = Counter(doc)
        N = len(doc)
        base[nama] = base["kata"].apply(lambda k: cnt.get(k, 0) / N if N > 0 else 0)

    base["df"] = base.apply(lambda r: sum(r[n] > 0 for n in nama_obat_list), axis=1)
    base["idf"] = np.log(len(nama_obat_list) / base["df"])

    for n in nama_obat_list:
        base[n] *= base["idf"]

    return base

tfidf_uni = build_tfidf(df["unigram"])
tfidf_bi = build_tfidf(df["bigram"])

# ===============================
# QUERY VECTOR
# ===============================
def proses_gejala(text):
    t = remove_stopwords(tokenize(cleaning(text)))

    g1 = Counter(ngram(t, 1))
    v1 = np.array([g1.get(k, 0) for k in tfidf_uni["kata"]]) * tfidf_uni["idf"].values

    g2 = Counter(ngram(t, 2))
    v2 = np.array([g2.get(k, 0) for k in tfidf_bi["kata"]]) * tfidf_bi["idf"].values

    return v1, v2

# ===============================
# COSINE SIMILARITY
# ===============================
def cari_obat(text, mode):
    v1, v2 = proses_gejala(text)
    v = v1 if mode == "unigram" else v2
    tfidf = tfidf_uni if mode == "unigram" else tfidf_bi

    hasil = []
    nv = np.linalg.norm(v)

    for nama in nama_obat_list:
        d = tfidf[nama].values
        nd = np.linalg.norm(d)
        skor = np.dot(v, d) / (nv * nd) if nv > 0 and nd > 0 else 0
        ket = df[df["Nama obat"] == nama]["Kegunaan obat"].values[0]
        hasil.append([nama, skor, ket])

    return pd.DataFrame(
        hasil, columns=["Nama Obat", "Skor", "Keterangan"]
    ).sort_values("Skor", ascending=False)

# ===============================
# GROUND TRUTH & PRECISION
# ===============================
def get_ground_truth(query):
    q = query.lower()
    row = gt_df[gt_df["query"].str.lower().str.contains(q)]
    if row.empty:
        return []
    obat = []
    for x in row["daftar_obat"]:
        obat.extend([o.strip() for o in x.split(",")])
    return list(set(obat))

def precision_at_3(df_hasil, gt):
    top3 = df_hasil.head(3)["Nama Obat"].tolist()
    relevan = [o for o in top3 if o in gt]
    return len(relevan) / 3

# ===============================
# OUTPUT
# ===============================
if run:
    col1, col2 = st.columns(2)
    gt = get_ground_truth(gejala)

    for col, mode, title in [
        (col1, "unigram", "UNIGRAM"),
        (col2, "bigram", "BIGRAM"),
    ]:
        with col:
            st.subheader(title)
            h = cari_obat(gejala, mode)

            if h["Skor"].max() == 0:
                st.info("Tidak ada kata yang cocok")
            else:
                if gt:
                    st.markdown(f"**Precision@3:** {precision_at_3(h, gt):.2f}")

                for _, r in h.head(3).iterrows():
                    st.markdown(
                        f"""
                        <div class="result-card">
                            <div class="obat-title">{r['Nama Obat']}</div>
                            <div class="score">Skor: {r['Skor']:.4f}</div>
                            <p>{r['Keterangan']}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
