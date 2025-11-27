import streamlit as st
import pandas as pd
import numpy as np
import re
from collections import Counter
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# ===============================
# ========== PAGE STYLE =========
# ===============================
st.set_page_config(page_title="Pencarian Obat - Skripsi", layout="wide")

st.markdown("""
<style>

body {
    background: white !important;
}

.search-box {
    padding: 20px;
    background: #87CEFA;   /* sky blue */
    border-radius: 14px;
    margin-bottom: 20px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.1);
}

.result-card {
    background: white;
    padding: 18px;
    border-radius: 16px;
    margin-bottom: 16px;
    box-shadow: 0 0 12px rgba(0, 136, 255, 0.35);   /* blue glow */
    border-left: 4px solid #0099ff;
}

.obat-title {
    font-weight: 700;
    font-size: 20px;
    color: #1b1b1b;
}

.score {
    font-size: 14px;
    color: #006eff;
    font-weight: bold;
}

.separator {
    border-left: 3px solid #dcdde1;
    height: 100%;
    margin: 0 25px;
}

</style>
""", unsafe_allow_html=True)

# ===============================
# ======== LOAD DATA ============
# ===============================

df = pd.read_csv("data_obat.csv")

stop_factory = StopWordRemoverFactory()
stopwords = set(stop_factory.get_stop_words())
tanda_baca = ['.', ',', '-', '–', ':', ';', '/', '(', ')', '®', '&']

# ===============================
# ======== PREPROCESS ===========
# ===============================

def cleaning(text):
    if pd.isna(text): 
        return ""
    text = re.sub(r"[0-9]", "", text)
    for ch in tanda_baca:
        text = text.replace(ch, "")
    return " ".join(text.split())

def case_folding(t): 
    return t.lower()

def tokenize(t): 
    return t.split()

def remove_stopwords(tokens): 
    return [w for w in tokens if w not in stopwords]

def buat_ngram(tokens, n):
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

df["clean"] = (
    df["Kegunaan obat"]
    .astype(str)
    .apply(cleaning)
    .apply(case_folding)
    .apply(tokenize)
    .apply(remove_stopwords)
)

df["unigram"] = df["clean"].apply(lambda t: buat_ngram(t,1))
df["bigram"]  = df["clean"].apply(lambda t: buat_ngram(t,2))

nama_obat_list = df["Nama obat"].tolist()

# ===============================
# ========= TF-IDF CORE =========
# ===============================

def build_tfidf(list_ngram):
    kata = sorted(list(set([w for doc in list_ngram for w in doc])))
    base = pd.DataFrame({"kata": kata})

    # TF
    for nama, doc in zip(nama_obat_list, list_ngram):
        count = Counter(doc)
        N = len(doc)
        base[nama] = base["kata"].apply(lambda k: count.get(k, 0) / N if N > 0 else 0)

    # DF
    base["df"] = base.apply(lambda r: sum(r[n] > 0 for n in nama_obat_list), axis=1)

    Ndoc = len(nama_obat_list)
    base["IDF"] = base["df"].apply(lambda d: np.log(Ndoc / d) if d > 0 else 0)

    # TF-IDF final
    tfidf = base.copy()
    for n in nama_obat_list:
        tfidf[n] = tfidf[n] * tfidf["IDF"]

    return tfidf

tfidf_uni = build_tfidf(df["unigram"])
tfidf_bi  = build_tfidf(df["bigram"])

# ===============================
# ======== GEJALA VECTOR ========
# ===============================

def proses_gejala(text):
    c = cleaning(text)
    c = case_folding(c)
    t = tokenize(c)
    s = remove_stopwords(t)

    # UNIGRAM
    g_uni = Counter(buat_ngram(s,1))
    N_uni = sum(g_uni.values())
    v_uni = np.array([g_uni.get(k, 0)/N_uni if N_uni > 0 else 0 for k in tfidf_uni["kata"]])
    v_uni = v_uni * tfidf_uni["IDF"].values

    # BIGRAM
    g_bi = Counter(buat_ngram(s,2))
    N_bi = sum(g_bi.values())
    v_bi = np.array([g_bi.get(k, 0)/N_bi if N_bi > 0 else 0 for k in tfidf_bi["kata"]])
    v_bi = v_bi * tfidf_bi["IDF"].values

    return v_uni, v_bi

# ===============================
# ========= CARI OBAT ===========
# ===============================

def cari_obat(text, mode="unigram"):
    v_uni, v_bi = proses_gejala(text)
    tfidf = tfidf_uni if mode == "unigram" else tfidf_bi
    v = v_uni if mode == "unigram" else v_bi

    norm_v = np.sqrt(np.sum(v * v))
    hasil = []

    for nama in nama_obat_list:
        d = tfidf[nama].values
        norm_d = np.sqrt(np.sum(d * d))
        dot = np.sum(v * d)
        score = dot / (norm_v * norm_d) if norm_v > 0 and norm_d > 0 else 0

        ket = df.loc[df["Nama obat"] == nama, "Kegunaan obat"].values[0]
        hasil.append([nama, score, ket])

    return pd.DataFrame(hasil, columns=["Nama Obat", "Skor", "Keterangan"]) \
             .sort_values("Skor", ascending=False)

# ===============================
# ============ UI ===============
# ===============================

st.title("🔍 Pencarian Obat Berdasarkan Gejala")
st.write("Masukkan gejala lalu sistem akan mencarikan obat paling relevan berdasarkan metode Unigram & Bigram.")

# ========== SEARCH BOX ==========
st.markdown('<div class="search-box">', unsafe_allow_html=True)
gejala = st.text_area("Masukkan gejala di sini:", height=100)
run = st.button("Cari Obat 🔎")
st.markdown('</div>', unsafe_allow_html=True)

# ========== OUTPUT ==========
if run and gejala.strip() != "":

    col_left, col_mid, col_right = st.columns([5, 0.3, 5])

    # ==========================
    #       UNIGRAM
    # ==========================
    with col_left:
        st.subheader("⚪ UNIGRAM RESULT")
        hasil_uni = cari_obat(gejala, "unigram").head(5)

        for _, row in hasil_uni.iterrows():
            st.markdown(f"""
            <div class="result-card">
                <div class="obat-title">{row['Nama Obat']}</div>
                <div class="score">Skor: {row['Skor']:.4f}</div>
                <p>{row['Keterangan']}</p>
            </div>
            """, unsafe_allow_html=True)

    # ==========================
    #     PEMBATAS TENGAH
    # ==========================
    with col_mid:
        st.markdown("<div class='separator'></div>", unsafe_allow_html=True)

    # ==========================
    #        BIGRAM
    # ==========================
    with col_right:
        st.subheader("🔵 BIGRAM RESULT")
        hasil_bi = cari_obat(gejala, "bigram").head(5)

        for _, row in hasil_bi.iterrows():
            st.markdown(f"""
            <div class="result-card">
                <div class="obat-title">{row['Nama Obat']}</div>
                <div class="score">Skor: {row['Skor']:.4f}</div>
                <p>{row['Keterangan']}</p>
            </div>
            """, unsafe_allow_html=True)
