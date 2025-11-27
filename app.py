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

# ===============================
#        LANDING PAGE UI
# ===============================

# inject CSS
st.markdown("""
<style>

.header-title {
    text-align: center;
    font-size: 32px;
    font-weight: 600;
    margin-top: -20px;
    margin-bottom: 20px;
    color: #1f2a37;
}

.search-wrapper {
    display: flex;
    justify-content: center;
    align-items: center;
    margin-top: -10px;
    margin-bottom: 20px;
}

.search-box-custom textarea {
    border: 2px solid #1A5F7A !important;
    border-radius: 30px !important;
    padding: 14px 20px !important;
    font-size: 16px !important;
}

.search-button {
    display: flex;
    justify-content: center;
    margin-top: -10px;
    margin-bottom: 20px;
}

.search-button button {
    background: white !important;
    color: #1A5F7A !important;
    border-radius: 25px !important;
    padding: 6px 30px !important;
    border: 2px solid #1A5F7A !important;
    font-size: 18px !important;
}

.search-button button:hover {
    background: #1A5F7A !important; 
    color: white !important;
    border-color: #1A5F7A !important;
}

.bg-image {
    position: absolute;
    bottom: 20px;
    right: 20px;
    opacity: 0.30;         /* transparansi 30% */
    z-index: -1;
}

</style>
""", unsafe_allow_html=True)


# ===============================
#        TAMPILAN AWAL
# ===============================

# judul
st.markdown("<div class='header-title'>PENCARIAN OBAT BERDASARKAN GEJALA</div>", unsafe_allow_html=True)

# gambar pojok kanan (bg-obat.png)
st.markdown(
    f"""
    <img src='data:image/png;base64,{open("bg-obat.png", "rb").read().encode("base64").decode()}' 
    class='bg-image' width='280'>
    """,
    unsafe_allow_html=True
)

# search box  
st.markdown("<div class='search-wrapper'>", unsafe_allow_html=True)
gejala = st.text_area(" ", placeholder="Tulis gejala anda disini", height=60, label_visibility="collapsed", key="searchbox")
st.markdown("</div>", unsafe_allow_html=True)

# tombol cari
st.markdown("<div class='search-button'>", unsafe_allow_html=True)
run = st.button("cari")
st.markdown("</div>", unsafe_allow_html=True)

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
        hasil_uni = cari_obat(gejala, "unigram").head(3)

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
        hasil_bi = cari_obat(gejala, "bigram").head(3)

        for _, row in hasil_bi.iterrows():
            st.markdown(f"""
            <div class="result-card">
                <div class="obat-title">{row['Nama Obat']}</div>
                <div class="score">Skor: {row['Skor']:.4f}</div>
                <p>{row['Keterangan']}</p>
            </div>
            """, unsafe_allow_html=True)
