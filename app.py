import streamlit as st
import pandas as pd
import numpy as np
import re
from collections import Counter
import base64
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

/* ==== BACKGROUND PAGE ==== */
[data-testid="stAppViewContainer"] {
    background: #F5F8F9 !important;   /* blue sky */
}


/* --- HEADER --- */
.header-title {
    text-align: center;
    font-size: 34px;
    font-weight: 600;
    margin-bottom: 30px;
    color: #1f2a37;
}

/* --- SEARCH BOX --- */
.search-box-custom textarea {
    border-radius: 40px !important;
    padding: 15px 22px !important;
    font-size: 17px !important;
    background: #8792AE !important;
}


/* --- BUTTON CARI --- */
.search-button button {
    background: #8792AE !important;
    color: #1A5F7A !important;
    border-radius: 30px !important;
    padding: 8px 35px !important;
    border: 2px solid #1A5F7A !important;
    font-size: 18px !important;
}


/* --- RESULT CARD --- */
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
#                       HEADER TITLE
# =========================================================
st.markdown("<div class='header-title'>PENCARIAN OBAT BERDASARKAN GEJALA PENYAKIT</div>", unsafe_allow_html=True)



# =========================================================
#                 SEARCH BOX (INPUT GEJALA) & BUTTON
# =========================================================
st.markdown("<div class='search-wrapper'>", unsafe_allow_html=True)
gejala = st.text_area("", height=60, label_visibility="collapsed")
st.markdown("</div>", unsafe_allow_html=True)

jumlah_kata = len(gejala.strip().split())

# validasi warna merah
if (jumlah_kata < 2 and gejala.strip() != "") or jumlah_kata > 15:
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

if jumlah_kata > 15:
    st.markdown("<div style='color:red'>Maksimal 15 kata</div>", unsafe_allow_html=True)

run = st.button(
    "Cari",
    disabled=(jumlah_kata < 2 or jumlah_kata > 15)
)
st.markdown("</div>", unsafe_allow_html=True)


# ============================
#     LIVE WORD COUNTER
# ============================

max_kata = 15
jumlah_word_input = len(gejala.strip().split())

# warna indikator
if jumlah_word_input < max_kata * 0.7:
    color = "green"
elif jumlah_word_input < max_kata:
    color = "orange"
else:
    color = "red"

st.markdown(
    f"<div style='text-align:center; margin-top:-10px; color:{color}; font-size:14px;'>"
    f"{jumlah_word_input}/{max_kata} kata"
    f"</div>",
    unsafe_allow_html=True
)

# stop jika input terlalu panjang
if jumlah_word_input > max_kata:
    st.error(f"input anda melebihi batas {max_kata} kata")
    st.stop()




# =========================================================
#                         LOAD DATASET
# =========================================================
df = pd.read_csv("data_obat.csv")

stop_factory = StopWordRemoverFactory()
stopwords = set(stop_factory.get_stop_words())
tanda_baca = ['.', ',', '-', '–', ':', ';', '/', '(', ')', '®', '&']


# =========================================================
#                     PREPROCESSING
# =========================================================
def cleaning(text):
    if pd.isna(text):
        return ""
    text = re.sub(r"[0-9]", "", text)
    for ch in tanda_baca:
        text = text.replace(ch, "")
    return " ".join(text.split())

def case_folding(t): return t.lower()
def tokenize(t): return t.split()
def remove_stopwords(tok): return [w for w in tok if w not in stopwords]

def ngram(tokens, n):
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

df["clean"] = (
    df["Kegunaan obat"].astype(str)
    .apply(cleaning).apply(case_folding)
    .apply(tokenize).apply(remove_stopwords)
)

df["unigram"] = df["clean"].apply(lambda t: ngram(t,1))
df["bigram"]  = df["clean"].apply(lambda t: ngram(t,2))

nama_obat_list = df["Nama obat"].tolist()


# =========================================================
#                        TF-IDF
# =========================================================
def build_tfidf(list_ngram):
    vocab = sorted(list(set([w for doc in list_ngram for w in doc])))
    base = pd.DataFrame({"kata": vocab})

    for nama, doc in zip(nama_obat_list, list_ngram):
        cnt = Counter(doc)
        N = len(doc)
        base[nama] = base["kata"].apply(lambda k: cnt.get(k,0)/N if N>0 else 0)

    base["df"] = base.apply(lambda r: sum(r[n] > 0 for n in nama_obat_list), axis=1)
    Ndoc = len(nama_obat_list)
    base["IDF"] = base["df"].apply(lambda d: np.log(Ndoc/d) if d>0 else 0)

    for n in nama_obat_list:
        base[n] = base[n] * base["IDF"]

    return base

tfidf_uni = build_tfidf(df["unigram"])
tfidf_bi  = build_tfidf(df["bigram"])


# =========================================================
#               GEJALA → VECTOR TF-IDF
# =========================================================
def proses_gejala(text):
    c = cleaning(text)
    t = tokenize(c)
    s = remove_stopwords(t)

    # unigram
    g1 = Counter(ngram(s,1))
    N1 = sum(g1.values())
    v1 = np.array([g1.get(k,0)/N1 if N1>0 else 0 for k in tfidf_uni["kata"]])
    v1 *= tfidf_uni["IDF"].values

    # bigram
    g2 = Counter(ngram(s,2))
    N2 = sum(g2.values())
    v2 = np.array([g2.get(k,0)/N2 if N2>0 else 0 for k in tfidf_bi["kata"]])
    v2 *= tfidf_bi["IDF"].values

    return v1, v2


# =========================================================
#                   CARI OBAT (COSINE)
# =========================================================
def cari_obat(text, mode="unigram"):
    v1, v2 = proses_gejala(text)
    v = v1 if mode == "unigram" else v2
    tfidf = tfidf_uni if mode == "unigram" else tfidf_bi

    nv = np.sqrt(np.sum(v*v))
    hasil = []

    for nama in nama_obat_list:
        d = tfidf[nama].values
        nd = np.sqrt(np.sum(d*d))
        dot = np.sum(v*d)
        skor = dot/(nv*nd) if nv>0 and nd>0 else 0

        ket = df[df["Nama obat"] == nama]["Kegunaan obat"].values[0]
        hasil.append([nama, skor, ket])

    return pd.DataFrame(hasil, columns=["Nama Obat", "Skor", "Keterangan"]) \
             .sort_values("Skor", ascending=False)


# =========================================================
#                        OUTPUT
# =========================================================
if run and gejala.strip() != "":

    col_left, col_mid, col_right = st.columns([5,0.3,5])

    # ======================
    #       UNIGRAM
    # ======================
    with col_left:
        st.subheader("♡ UNIGRAM RESULT")
        hasil_uni = cari_obat(gejala, "unigram")

        if hasil_uni["Skor"].max() == 0:
            st.info("Tidak ada kata yang cocok")
        else:
            for _, row in hasil_uni.head(3).iterrows():
                st.markdown(f"""
                <div class="result-card">
                    <div class="obat-title">{row['Nama Obat']}</div>
                    <div class="score">Skor: {row['Skor']:.4f}</div>
                    <p>{row['Keterangan']}</p>
                </div>
                """, unsafe_allow_html=True)

    # ======================
    #        BIGRAM
    # ======================
    with col_right:
        st.subheader("♡ BIGRAM RESULT")
        hasil_bi = cari_obat(gejala, "bigram")

        if hasil_bi["Skor"].max() == 0:
            st.info("Tidak ada kata yang cocok")
        else:
            for _, row in hasil_bi.head(3).iterrows():
                st.markdown(f"""
                <div class="result-card">
                    <div class="obat-title">{row['Nama Obat']}</div>
                    <div class="score">Skor: {row['Skor']:.4f}</div>
                    <p>{row['Keterangan']}</p>
                </div>
                """, unsafe_allow_html=True)

