import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter
from collections import defaultdict
import pickle
import pandas as pd
import io
import gc

st.set_page_config(page_title="Sound Sniff", layout="wide")
st.title("🎵 Sound Sniff: The Audio Detective")

@st.cache_data
def load_database():
    with open("song_database_small.pkl", "rb") as f:
        return pickle.load(f)

db = load_database()
st.sidebar.success(f"✅ Database loaded! ({len(db)} unique hashes)")

def get_peaks_and_hashes(y, sr, hop_length=512, threshold=-20, fan_value=5):
    D          = librosa.stft(y, hop_length=hop_length)
    S_db       = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    del D
    gc.collect()
    local_max  = maximum_filter(S_db, size=20) == S_db
    peaks      = local_max & (S_db > threshold)
    peak_freqs, peak_times = np.where(peaks)
    hashes         = []
    sorted_indices = np.argsort(peak_times)
    s_freqs        = peak_freqs[sorted_indices]
    s_times        = peak_times[sorted_indices]
    for i in range(len(s_freqs) - fan_value):
        for j in range(1, fan_value + 1):
            dt = s_times[i+j] - s_times[i]
            hashes.append(((s_freqs[i], s_freqs[i+j], dt), s_times[i]))
    return S_db, peak_freqs, peak_times, hashes

def match_song(query_hashes, db):
    match_tally  = defaultdict(int)
    offset_tally = defaultdict(list)
    for query_hash, query_t1 in query_hashes:
        if query_hash in db:
            for db_song_name, db_t1 in db[query_hash]:
                offset = db_t1 - query_t1
                match_tally[(db_song_name, offset)] += 1
                offset_tally[db_song_name].append(offset)
    if not match_tally:
        return "No Match Found", None, None
    best = max(match_tally, key=match_tally.get)
    return best[0], best[1], offset_tally[best[0]]

tab1, tab2 = st.tabs(["🎵 Single-Clip Mode", "📂 Batch Mode"])

with tab1:
    st.header("Identify a Single Clip")
    uploaded = st.file_uploader("Upload a clip (.mp3 or .wav)",
                                 type=["mp3","wav"])
    if uploaded:
        st.audio(uploaded)
        with st.spinner("Fingerprinting audio..."):
            # Load at lower sample rate to save memory
            y, sr = librosa.load(io.BytesIO(uploaded.read()),
                                  sr=22050,       # reduced from 22050
                                  mono=True,
                                  duration=60)    # max 60 seconds
            gc.collect()
            S_db, peak_freqs, peak_times, q_hashes = get_peaks_and_hashes(y, sr)
            predicted, best_offset, all_offsets    = match_song(q_hashes, db)
            gc.collect()

        st.success(f"🎶 Matched Song: **{predicted}**")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Spectrogram + Constellation")
            fig, ax = plt.subplots(figsize=(8, 4))
            librosa.display.specshow(S_db, sr=sr, hop_length=512,
                                     x_axis="time", y_axis="linear",
                                     cmap="viridis", ax=ax)
            freq_res = sr / 2048
            ax.scatter(peak_times * 512/sr, peak_freqs * freq_res,
                       color="red", s=8, label="Peaks")
            ax.set_ylim(0, 4000)
            ax.legend(fontsize=8)
            st.pyplot(fig)
            plt.close()
            gc.collect()

        with col2:
            st.subheader("Offset Histogram")
            if all_offsets:
                fig2, ax2 = plt.subplots(figsize=(8, 4))
                ax2.hist(all_offsets, bins=50,
                         color="coral", edgecolor="black")
                ax2.set_xlabel("Time Offset (frames)")
                ax2.set_ylabel("Matching Hash Count")
                ax2.set_title(f"Alignment: {predicted}")
                st.pyplot(fig2)
                plt.close()
                gc.collect()

with tab2:
    st.header("Batch Processing")
    batch = st.file_uploader("Upload multiple clips",
                              type=["mp3","wav"],
                              accept_multiple_files=True)
    if st.button("🚀 Run Batch") and batch:
        results = []
        bar = st.progress(0)
        for i, clip in enumerate(batch):
            y, sr = librosa.load(io.BytesIO(clip.read()),
                                  sr=22050,
                                  mono=True,
                                  duration=60)
            _, _, _, q_hashes = get_peaks_and_hashes(y, sr)
            pred, *_ = match_song(q_hashes, db)
            results.append({"filename": clip.name,
                            "prediction": pred})
            bar.progress((i+1)/len(batch))
            gc.collect()

        df = pd.DataFrame(results)
        st.dataframe(df)
        st.download_button("⬇️ Download results.csv",
                           df.to_csv(index=False).encode(),
                           "results.csv", "text/csv")
