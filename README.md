# 💬 Chatbot template

A simple Streamlit app that shows how to build a chatbot using OpenAI's GPT-3.5.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://chatbot-template.streamlit.app/)

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```

---

## Custom Feature: 中文歌詞聚類 Chatbot

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

### ✅ 測試方式

打開終端機並執行：

```bash
streamlit run streamlit_app.py