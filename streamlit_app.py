import streamlit as st
import time
import re
import base64
from io import BytesIO
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import json
from collections import Counter
from typing import Union

placeholderstr = "Please input your command"
user_name = "譽心"
user_image = "https://www.w3schools.com/howto/img_avatar.png"

def stream_data(stream_str):
    for word in stream_str.split(" "):
        yield word + " "
        time.sleep(0.15)

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

    # Load data
    with open("full_record.json", "r", encoding="utf-8") as f:
        full_record = json.load(f)

    # Load stopwords
    with open("stopwords.txt", "r", encoding="utf-8") as f:
        stopwords = set([line.strip() for line in f if line.strip()])
    custom_stopwords = {"(",")","Studios", "混音", ":", "!", "/", "...", ".", ",", "'", "Studio", "工程 師", "BY2", "工程師"}
    stopwords = stopwords.union(custom_stopwords)

    # Clean lyrics
    def clean_and_tokenize_lyrics(track_record):
        for track in track_record:
            lyric = track["Lyrics"]
            tokens = jieba.lcut(lyric)
            clean_tokens = [word for word in tokens if word not in stopwords and word.strip()]
            track["Tokens"] = clean_tokens
        documents = [' '.join(track["Tokens"]) for track in track_record]
        return documents

    def clean_and_tokenize_input(text: str) -> str:
        tokens = jieba.lcut(text)
        clean_tokens = [word for word in tokens if word not in stopwords and word.strip()]
        return ' '.join(clean_tokens)

    def predict_cluster(text: str) -> str:
        cleaned = clean_and_tokenize_input(text)
        print(f"[DEBUG] 斷詞結果: {cleaned}")

        vec = vectorizer.transform([cleaned])
        print(f"[DEBUG] 向量維度: {vec.shape}")
        print(f"[DEBUG] 非零詞數 (nnz): {vec.nnz}")

        cluster_id = kmeans.predict(vec)[0]
        print(f"[DEBUG] 預測群號: {cluster_id}")

        return f"這段歌詞屬於第 {cluster_id} 群"


    from collections import Counter

    def generate_wordcloud_image(text: str = None, tokens: list[str] = None, title: str = None) -> Union[BytesIO, str]:
        """
        產生詞雲圖像，可以傳入原始文字（text）或已斷詞的 tokens。
        若無有效詞彙，回傳警告字串，避免 ValueError。
        """
        # 若未提供 tokens 則從 text 自動斷詞
        if tokens is None:
            if not text:
                return "⚠️ 沒有輸入文字內容。"
            tokens = jieba.lcut(text)

        # 移除停用詞
        clean_tokens = [word for word in tokens if word not in stopwords and word.strip()]
        
        if not clean_tokens:
            return "⚠️ 沒有有效的詞彙可產生詞雲，請確認輸入或群號是否正確。"

        freq_dict = dict(Counter(clean_tokens))

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


    def show_cluster_wordcloud(cluster_id: int):
        # 判斷目前群的範圍
        cluster_ids = set(track.get("Cluster") for track in full_record if "Cluster" in track)
        max_cluster_id = max(cluster_ids)

        if cluster_id > max_cluster_id or cluster_id < 0:
            return f"⚠️ 目前共有 {max_cluster_id + 1} 群，找不到第 {cluster_id} 群，請確認群號是否正確。"

        # 收集該群的 tokens
        cluster_tokens = []
        for track in full_record:
            if track.get("Cluster") == cluster_id and "Tokens" in track:
                cluster_tokens.extend(track["Tokens"])

        if not cluster_tokens:
            return f"⚠️ 第 {cluster_id} 群中沒有有效的詞彙可產生詞雲。"

        # 傳入 tokens 而不是 text，避免重新斷詞錯誤
        return generate_wordcloud_image(tokens=cluster_tokens, title=f"第 {cluster_id} 群 詞雲")

    def summarize_clusters() -> str:
        summary = []
        cluster_counts = {}
        token_counts = {}

        for track in full_record:
            cid = track.get("Cluster")
            if cid is None:
                continue
            cluster_counts[cid] = cluster_counts.get(cid, 0) + 1
            token_counts[cid] = token_counts.get(cid, 0) + len(track.get("Tokens", []))

        for cid in sorted(cluster_counts):
            summary.append(f"🔹 第 {cid} 群：{cluster_counts[cid]} 首歌，{token_counts[cid]} 個詞")

        if not summary:
            return "⚠️ 尚未完成分群，請確認模型已訓練。"

        return "📊 分群摘要：\n" + "\n".join(summary)


    def generate_response(prompt):
        prompt = prompt.strip()
        print(f"[DEBUG] 使用者輸入：{prompt}")

        # 群詞雲邏輯放最前，避免被 "畫...詞雲" 攔截
        if re.search(r"第\s*\d+\s*群.*詞雲", prompt):
            print("[DEBUG] 進入群詞雲顯示邏輯")
            match = re.search(r"第\s*(\d+)\s*群", prompt)
            if match:
                cluster_id = int(match.group(1))
                return show_cluster_wordcloud(cluster_id)
            else:
                return "請提供有效的群號（例如：第 2 群）。"

        elif re.search(r"(畫|生成|給我).*詞雲", prompt):
            print("[DEBUG] 進入詞雲生成邏輯")
            text = prompt.split('詞雲')[-1].strip(":： ")
            return generate_wordcloud_image(text)

        elif "分類" in prompt or "這段歌詞屬於哪一群" in prompt:
            print("[DEBUG] 進入分群預測邏輯")
            text = prompt.split("歌詞")[-1].strip(":： ")
            return predict_cluster(text)

        elif "幾群" in prompt or "目前有幾群" in prompt or "群摘要" in prompt:
            print("[DEBUG] 進入分群摘要邏輯")
            return summarize_clusters()

        else:
            print("[DEBUG] 落入 fallback 回傳區")
            return f"You say: {prompt}."


    def chat(prompt: str):
        st_c_chat.chat_message("user", avatar=user_image).write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        response = generate_response(prompt)
        st.session_state.messages.append({"role": "assistant", "content": response})

        if isinstance(response, BytesIO):
            st_c_chat.chat_message("assistant").image(response)
        else:
            st_c_chat.chat_message("assistant").write_stream(stream_data(response))

    # Prepare model
    cleaned_docs = clean_and_tokenize_lyrics(full_record)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(cleaned_docs)
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans.fit(tfidf_matrix)

    labels = kmeans.labels_
    for i, track in enumerate(full_record):
        track["Cluster"] = int(labels[i])

    if prompt := st.chat_input(placeholder=placeholderstr, key="chat_bot"):
        chat(prompt)

if __name__ == "__main__":
    main()
