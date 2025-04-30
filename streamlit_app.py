import streamlit as st
import time
import re
import base64
import os
from io import BytesIO
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import json
from collections import Counter
from typing import Union
from gensim.utils import simple_preprocess
from gensim.models import Word2Vec
import numpy as np
import plotly.graph_objs as go
from sklearn.decomposition import PCA
import matplotlib.colors as mcolors
import matplotlib.cm as cm



placeholderstr = "Please input your command"
user_name = "O_o"
user_image = "https://www.w3schools.com/howto/img_avatar.png"

def stream_data(stream_str):
    for word in stream_str.split(" "):
        yield word + " "
        time.sleep(0.15)

# ========== 可重用功能 ==============
def preprocess_input(text):
    sentences = text.strip().split("\n")
    return [simple_preprocess(s) for s in sentences if s.strip()]

def train_word2vec(tokenized_sentences, sg, vector_size=100, window=5):
    return Word2Vec(tokenized_sentences, vector_size=vector_size, window=window, min_count=1, workers=4, seed=42, sg=sg)

def get_distinct_colors(n, cmap_name='tab20'):
    cmap = cm.get_cmap(cmap_name, n)
    return [mcolors.to_hex(cmap(i)) for i in range(n)]

# ========== General（分群） ==============
class General:
    def __init__(self):
        with open("full_record.json", "r", encoding="utf-8") as f:
            self.full_record = json.load(f)

        with open("stopwords.txt", "r", encoding="utf-8") as f:
            stopwords = set([line.strip() for line in f if line.strip()])
        custom_stopwords = {"(", ")", "Studios", "混音", ":", "!", "/", "...", ".", ",", "'", "Studio", "工程 師", "BY2", "工程師"}
        self.stopwords = stopwords.union(custom_stopwords)

        cleaned_docs = self.clean_and_tokenize_lyrics(self.full_record)
        self.vectorizer = TfidfVectorizer()
        tfidf_matrix = self.vectorizer.fit_transform(cleaned_docs)
        self.kmeans = KMeans(n_clusters=4, random_state=42)
        self.kmeans.fit(tfidf_matrix)

        labels = self.kmeans.labels_
        for i, track in enumerate(self.full_record):
            track["Cluster"] = int(labels[i])

    def clean_and_tokenize_lyrics(self, track_record):
        for track in track_record:
            lyric = track["Lyrics"]
            tokens = jieba.lcut(lyric)
            clean_tokens = [word for word in tokens if word not in self.stopwords and word.strip()]
            track["Tokens"] = clean_tokens
        return [' '.join(track["Tokens"]) for track in track_record]

    def clean_and_tokenize_input(self, text: str) -> str:
        tokens = jieba.lcut(text)
        clean_tokens = [word for word in tokens if word not in self.stopwords and word.strip()]
        return ' '.join(clean_tokens)

    def predict_cluster(self, text: str) -> str:
        cleaned = self.clean_and_tokenize_input(text)
        vec = self.vectorizer.transform([cleaned])
        cluster_id = self.kmeans.predict(vec)[0]
        return f"這段歌詞屬於第 {cluster_id} 群"

    def safe_wordcloud(freq_dict, title=None):
        try:
            wc = WordCloud(
                width=1500,
                height=1500,
                background_color='white',
                max_words=200,
                font_path="TaipeiSansTCBeta-Regular.ttf",
                random_state=50,
                contour_width=1,
                contour_color='black',
                colormap='copper',
                prefer_horizontal=0.9
            )
            wc.generate_from_frequencies(freq_dict)
        except OSError:
            wc = WordCloud(
                width=1500,
                height=1500,
                background_color='white',
                max_words=200,
                random_state=50,
                contour_width=1,
                contour_color='black',
                colormap='copper',
                prefer_horizontal=0.9
            )
            wc.generate_from_frequencies(freq_dict)

        fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        if title:
            ax.set_title(title, fontsize=24, pad=10)
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return buf
    
    def generate_wordcloud_image(self, text: str = None, tokens: list[str] = None, title: str = None) -> Union[BytesIO, str]:
        if tokens is None:
            if not text:
                return "⚠️ 沒有輸入文字內容。"
            tokens = jieba.lcut(text)

        clean_tokens = [word for word in tokens if word not in self.stopwords and word.strip()]
        if not clean_tokens:
            return "⚠️ 沒有有效的詞彙可產生詞雲，請確認輸入或群號是否正確。"

        freq_dict = dict(Counter(clean_tokens))
        return General.safe_wordcloud(freq_dict, title=title)

    def show_cluster_wordcloud(self, cluster_id: int):
        cluster_ids = set(track.get("Cluster") for track in self.full_record if "Cluster" in track)
        max_cluster_id = max(cluster_ids)
        if cluster_id > max_cluster_id or cluster_id < 0:
            return f"⚠️ 目前共有 {max_cluster_id + 1} 群，找不到第 {cluster_id} 群，請確認群號是否正確。"
        cluster_tokens = []
        for track in self.full_record:
            if track.get("Cluster") == cluster_id and "Tokens" in track:
                cluster_tokens.extend(track["Tokens"])
        if not cluster_tokens:
            return f"⚠️ 第 {cluster_id} 群中沒有有效的詞彙可產生詞雲。"
        return self.generate_wordcloud_image(tokens=cluster_tokens, title=f"第 {cluster_id} 群 詞雲")

    def summarize_clusters(self) -> str:
        summary = []
        cluster_counts = {}
        token_counts = {}
        for track in self.full_record:
            cid = track.get("Cluster")
            if cid is None:
                continue
            cluster_counts[cid] = cluster_counts.get(cid, 0) + 1
            token_counts[cid] = token_counts.get(cid, 0) + len(track.get("Tokens", []))
        for cid in sorted(cluster_counts):
            summary.append(f"\n🔹 第 {cid} 群：{cluster_counts[cid]} 首歌，{token_counts[cid]} 個詞")
        if not summary:
            return "⚠️ 尚未完成分群，請確認模型已訓練。"
        return "📊 分群摘要：\n" + "\n".join(summary)

    def generate_response(self, prompt):
        prompt = prompt.strip()
        if re.search(r"第\s*\d+\s*群.*詞雲", prompt):
            match = re.search(r"第\s*(\d+)\s*群", prompt)
            if match:
                cluster_id = int(match.group(1))
                return self.show_cluster_wordcloud(cluster_id)
            else:
                return "請提供有效的群號（例如：第 2 群）。"
        elif re.search(r"(畫|生成|給我).*詞雲", prompt):
            text = prompt.split('詞雲')[-1].strip(":： ")
            return self.generate_wordcloud_image(text)
        elif "分類" in prompt or "這段歌詞屬於哪一群" in prompt:
            text = prompt.split("歌詞")[-1].strip(":： ")
            return self.predict_cluster(text)
        elif "幾群" in prompt or "目前有幾群" in prompt or "群摘要" in prompt:
            return self.summarize_clusters()
        else:
            return f"You say: {prompt}."


# ========== Q1-1 2D ==============
def visualize_2d(model, tokenized_sentences):
    word_vectors = np.array([model.wv[word] for word in model.wv.index_to_key])
    pca_2d = PCA(n_components=2).fit_transform(word_vectors)
    color_map = get_distinct_colors(len(tokenized_sentences))

    word_colors = []
    for word in model.wv.index_to_key:
        for i, sentence in enumerate(tokenized_sentences):
            if word in sentence:
                word_colors.append(color_map[i % len(color_map)])
                break

    scatter = go.Scatter(
        x=pca_2d[:, 0],
        y=pca_2d[:, 1],
        mode='markers+text',
        text=model.wv.index_to_key,
        textposition='top center',
        marker=dict(color=word_colors, size=8),
        hovertemplate="Word: %{text}"
    )

    line_traces = []
    for i, sentence in enumerate(tokenized_sentences):
        vectors = [pca_2d[model.wv.key_to_index[word]] for word in sentence if word in model.wv.key_to_index]
        line_trace = go.Scatter(
            x=[v[0] for v in vectors],
            y=[v[1] for v in vectors],
            mode='lines',
            line=dict(color=color_map[i % len(color_map)], width=1),
            showlegend=True,
            name=f"Sentence {i+1}"
        )
        line_traces.append(line_trace)

    fig = go.Figure(data=[scatter] + line_traces)
    fig.update_layout(title="2D Word Embedding Visualization", width=1000, height=1000)
    return fig

# ========== Q1-1 3D ==============

def visualize_3d(model, tokenized_sentences):
    word_vectors = np.array([model.wv[word] for word in model.wv.index_to_key])
    pca_3d = PCA(n_components=3).fit_transform(word_vectors)
    color_map = get_distinct_colors(len(tokenized_sentences))

    word_colors = []
    for word in model.wv.index_to_key:
        for i, sentence in enumerate(tokenized_sentences):
            if word in sentence:
                word_colors.append(color_map[i % len(color_map)])
                break

    scatter = go.Scatter3d(
        x=pca_3d[:, 0],
        y=pca_3d[:, 1],
        z=pca_3d[:, 2],
        mode='markers+text',
        text=model.wv.index_to_key,
        textposition='top center',
        marker=dict(color=word_colors, size=2)
    )

    fig = go.Figure(data=[scatter])
    fig.update_layout(
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
        title="3D Word Embedding Visualization",
        width=1000,
        height=1000
    )
    return fig

# ========== Q2 Skip-gram ==============
def run_skipgram_analysis(tokenized_sentences, keyword=None):
    model = train_word2vec(tokenized_sentences, sg=1)
    if keyword is None:
        keyword = model.wv.index_to_key[0]
    if keyword not in model.wv:
        return f"⚠️ 詞彙 '{keyword}' 不在語料中。"
    vector = model.wv[keyword]
    similar = model.wv.most_similar(keyword)
    return vector, similar

# ========== Q3 CBOW ==============
def run_cbow_analysis(tokenized_sentences, keyword=None):
    model = train_word2vec(tokenized_sentences, sg=0)
    if keyword is None:
        keyword = model.wv.index_to_key[0]
    if keyword not in model.wv:
        return f"⚠️ 詞彙 '{keyword}' 不在語料中。"
    vector = model.wv[keyword]
    similar = model.wv.most_similar(keyword)
    return vector, similar

def main():
    st.set_page_config(
        page_title='K-Assistant - The Residemy Agent',
        layout='wide',
        initial_sidebar_state='auto',
        menu_items={
            'Get Help': 'https://streamlit.io/',
            'Report a bug': 'https://github.com',
            'About': 'About your application: **Hello world**'
        },
        page_icon="img/favicon.ico"
    )

    st.title(f"💬 {user_name}'s Chatbot")

    with st.sidebar:
        selected_lang = st.selectbox("Language", ["English", "繁體中文"], index=1)
        lang_setting = st.session_state.get('lang_setting', selected_lang)
        st.session_state['lang_setting'] = lang_setting

        st_c_1 = st.container(border=True)
        with st_c_1:
            st.image("https://www.w3schools.com/howto/img_avatar.png")

        selected_page = st.radio("功能選擇", ["General", "Q1-1 2D", "Q1-1 3D", "Q2 SKIP-GRAM", "Q3 CBOW"])

    st_c_chat = st.container(border=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    else:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st_c_chat.chat_message(msg["role"], avatar=user_image).markdown(msg["content"])
            elif msg["role"] == "assistant":
                st_c_chat.chat_message(msg["role"]).markdown(msg["content"])
            else:
                try:
                    image_tmp = msg.get("image")
                    if image_tmp:
                        st_c_chat.chat_message(msg["role"], avatar=image_tmp).markdown(msg["content"])
                except:
                    st_c_chat.chat_message(msg["role"]).markdown(msg["content"])

    if selected_page == "General":
        if "general" not in st.session_state:
            st.session_state.general = General()
        general = st.session_state.general

        if prompt := st.chat_input(placeholder=placeholderstr, key="chat_bot"):
            st_c_chat.chat_message("user", avatar=user_image).write(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            response = general.generate_response(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response})
            if isinstance(response, BytesIO):
                st_c_chat.chat_message("assistant").image(response)
            else:
                st_c_chat.chat_message("assistant").write_stream(stream_data(response))
    elif selected_page == "Q1-1 2D":
        st.subheader("📊 Q1-1: 2D Visualization")
        prompt = st.chat_input("Please input sentences, e.g., He forgot his umbrella on the crowded subway.", key="q1_2d_input")
        if prompt:
            tokenized = preprocess_input(prompt)
            model = train_word2vec(tokenized, sg=1)
            fig = visualize_2d(model, tokenized)
            st.plotly_chart(fig)

    elif selected_page == "Q1-1 3D":
        st.subheader("📊 Q1-1: 3D Visualization")
        prompt = st.chat_input("Please input sentences, e.g., He forgot his umbrella on the crowded subway.", key="q1_3d_input")
        if prompt:
            tokenized = preprocess_input(prompt)
            model = train_word2vec(tokenized, sg=1)
            fig = visualize_3d(model, tokenized)
            st.plotly_chart(fig)

    elif selected_page == "Q2 SKIP-GRAM":
        st.subheader("🧠 Q2: Skip-gram")

        # 第一步：輸入語料
        if "q2_tokenized" not in st.session_state:
            prompt = st.chat_input("Please input sentences to analyze.", key="q2_input")
            if prompt:
                tokenized = preprocess_input(prompt)
                st.session_state.q2_tokenized = tokenized
                st.rerun()

        # 第二步：輸入要探索的詞
        elif "q2_tokenized" in st.session_state:
            keyword = st.chat_input("‼️Please input the keyword you wanna know more‼️", key="q2_keyword")
            if keyword:
                result = run_skipgram_analysis(st.session_state.q2_tokenized, keyword)
                if isinstance(result, str):
                    st.warning(result)
                else:
                    vector, similar = result
                    st.write("📌 Vector for the keyword：", vector)
                    st.write("🔍 Most similar words to the keyword ：", similar)

            # 加入 reset 按鈕讓使用者重來
            if st.button("🔁 Input new sentences to analyze."):
                del st.session_state.q2_tokenized
                st.rerun()

    elif selected_page == "Q3 CBOW":
        st.subheader("📘 Q3: CBOW")

        # 第一步：輸入語料
        if "q3_tokenized" not in st.session_state:
            prompt = st.chat_input("Please input sentences to analyze.", key="q3_input")
            if prompt:
                tokenized = preprocess_input(prompt)
                st.session_state.q3_tokenized = tokenized
                st.rerun()

        # 第二步：輸入要探索的詞
        elif "q3_tokenized" in st.session_state:
            keyword = st.chat_input("‼️Please input the keyword you wanna know more‼️", key="q3_keyword")
            if keyword:
                result = run_cbow_analysis(st.session_state.q3_tokenized, keyword)
                if isinstance(result, str):
                    st.warning(result)
                else:
                    vector, similar = result
                    st.write("📌 Vector for the keyword：", vector)
                    st.write("🔍 Most similar words to the keyword ：", similar)

            # 加入 reset 按鈕讓使用者重來
            if st.button("🔁 Input new sentences to analyze."):
                del st.session_state.q3_tokenized
                st.rerun()


if __name__ == "__main__":
    main()
