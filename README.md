# 🎵 Lyrics Mining Agent

An interactive chatbot that analyzes Chinese lyrics using clustering and word cloud generation, and supports English input for Word2Vec-based semantic visualization. Built with Streamlit and extended from a course project originally designed to demonstrate OpenAI GPT-3.5 integration.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://chatbot-template.streamlit.app/)

---

## 🧠 About This Project

This project extends a text mining course template into a fully interactive AI agent. It enables users to input Mandarin song lyrics and receive:

- Cluster prediction using KMeans
- Word cloud generation based on lyric content
- Summary statistics of lyric clusters
- Semantic visualization using English Word2Vec embeddings

This app showcases how clustering, embeddings, and NLP interaction can be integrated into a bilingual chatbot built with Streamlit.

---

## How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```

---

## 中文歌詞聚類 Chatbot

這個 chatbot 整合了詞雲產生與 KMeans 聚類的功能，能即時回答與歌詞相關的互動式提問。

### 功能一覽

| 功能類型         | 指令範例                                       | 回應類型     |
|------------------|------------------------------------------------|--------------|
|  詞雲產生       | 畫出這段歌詞的詞雲：我想你了                    | 圖片（詞雲） |
|  分群預測       | 分類這段歌詞：我們笑著說再見，心裡卻泛著淚       | 文字         |
|  群詞雲顯示     | 請給我第 1 群的詞雲                              | 圖片（詞雲） |
|  分群摘要統計   | 群摘要 / 目前有幾群？                            | 文字         |
|  群不存在時提醒 | 請給我第 999 群的詞雲                            | 警告訊息     |

---

## Word Embedding & Semantic Visualization (English Input)

These tabs support English input to analyze and visualize word relationships using Word2Vec models trained on user-defined input.

### Q1-1, 1-2: Word Embedding Visualization

| Feature                  | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| 2D Visualization         | Input 1+ English sentences and view token relationships using PCA (2D)     |
| 3D Visualization         | Similar to above, but with 3D interactive scatter plot                     |

### Q2: Skip-gram Analysis

| Feature                    | Description                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| Step 1: Sentence Input     | Input English sentences (multi-line supported)                             |
| Step 2: Keyword Query      | Input a target word to get its vector and most similar words (Skip-gram)   |
| Reset Input                | Click the button to input a new sentence corpus                            |

### Q3: CBOW Analysis

| Feature                    | Description                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| Step 1: Sentence Input     | Input English sentences (multi-line supported)                             |
| Step 2: Keyword Query      | Input a target word to get its vector and most similar words (CBOW)        |
| Reset Input                | Click the button to input a new sentence corpus                            |

> Note: The Word2Vec model is trained live on your input each time.

---

## 🙋 Author's Note

This chatbot was developed as part of a text mining midterm project and enhanced with original features including lyric clustering, multilingual support, and dynamic visualization.  
Forked from [GildShen/Gild-chatbot](https://github.com/GildShen/Gild-chatbot) and significantly extended for advanced lyric analysis and interactive AI agent functionality.