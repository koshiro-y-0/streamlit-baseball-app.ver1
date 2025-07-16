
'''Anaconda Prompt でだけ Streamlit'''
# cd Desktop/"Github(DeNA)"
# streamlit run dena_2024_code.py

'''Streamlitで実行'''
# streamlit run dena_2024_code.py


import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import requests
from bs4 import BeautifulSoup

# 日本語フォント設定
matplotlib.rcParams['font.family'] = 'Yu Gothic'

def preprocess(df):
    df["年度"] = pd.to_numeric(df["年度"], errors="coerce").astype("Int64")
    for col in ["打率", "OPS", "本塁打"]:
        df[col] = df[col].astype(str).str.strip()
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["打率", "OPS", "本塁打"])
    df["MVPスコア"] = df["打率"] * df["OPS"] * df["本塁打"]
    return df

# CSV読み込み
df_2024_csv = pd.read_csv("dena_2024_stats.csv")
df_2025_csv = pd.read_csv("dena_2025_stats.csv")

df_2024 = preprocess(df_2024_csv)
df_2025 = preprocess(df_2025_csv)

# ✅ 結合
df = pd.concat([df_2024, df_2025], ignore_index=True)

# アプリ説明
st.markdown("""
# 横浜DeNAベイスターズ 打者成績分析アプリ ⚾  
このアプリでは **2023年と2024年の打者成績** を比較・可視化できます。  
選手ごとの成績グラフやMVPスコアランキングも表示されます。
""")

st.title("横浜DeNAベイスターズ 打者成績分析")

# 年度別の成績表示
st.subheader("📅 2023年 打者成績")
df_2023 = df[df["年度"] == 2023]
st.dataframe(df_2023[["名前", "打率", "本塁打", "出塁率", "OPS"]])

st.subheader("📅 2024年 打者成績")
df_2024 = df[df["年度"] == 2024]
st.dataframe(df_2024[["名前", "打率", "本塁打", "出塁率", "OPS"]])

# 選手選択
selected_player = st.selectbox("選手を選んでください", sorted(df["名前"].unique()), key="latest_select")

# 最新年取得 & データ抽出
latest_year = df[df["名前"] == selected_player]["年度"].max()
player_data = df[(df["名前"] == selected_player) & (df["年度"] == latest_year)].iloc[0]

# タイトル
st.subheader(f"📊 最新の打撃成績（{latest_year}年）")

# 年度別に成績を縦表示
player_df_pivot = df[df["名前"] == selected_player]

# 表示用に小数フォーマットを整える（本塁打は整数）
player_df_formatted = player_df_pivot.copy()
for col in ["打率", "出塁率", "OPS"]:
    player_df_formatted[col] = pd.to_numeric(player_df_formatted[col], errors="coerce").map("{:.3f}".format)
player_df_formatted["本塁打"] = pd.to_numeric(player_df_formatted["本塁打"], errors="coerce").astype("Int64")

# ✅ 最新年度だけ表示
latest_player_row = player_df_formatted[player_df_formatted["年度"] == latest_year]
latest_player_row = latest_player_row[["年度", "打率", "本塁打", "出塁率", "OPS"]].sort_values("年度")

st.table(latest_player_row)

# 棒グラフ描画
labels = ["打率", "出塁率", "OPS"]
values = [
    float(player_data["打率"]) if pd.notna(player_data["打率"]) else 0.0,
    float(player_data["出塁率"]) if pd.notna(player_data["出塁率"]) else 0.0,
    float(player_data["OPS"]) if pd.notna(player_data["OPS"]) else 0.0,
]
fig, ax = plt.subplots()
ax.bar(labels, values, color=["skyblue", "orange", "green"])
ax.set_ylim(0, 1.0)
st.pyplot(fig)

# MVPスコアランキング
st.subheader("🏆 MVPスコアランキング")
ranked_df = df.sort_values("MVPスコア", ascending=False)
st.bar_chart(ranked_df.set_index("名前")["MVPスコア"])

# 年度比較用
selected_years = st.multiselect("比較したい年度を選んでください", df["年度"].unique(), default=[2023, 2024])
selected_player_pivot = st.selectbox("選手を選んでください（年度横並び比較）", sorted(df["名前"].unique()))
player_df = df[df["名前"] == selected_player_pivot]

# ピボットテーブル整形
pivot_df = player_df.pivot_table(
    index="名前",
    columns="年度",
    values=["打率", "本塁打", "出塁率", "OPS"],
    aggfunc="first"
)
pivot_df.columns = [f"{metric}_{year}" for metric, year in pivot_df.columns]
pivot_df.reset_index(inplace=True)
cols = [
    "名前",
    "打率_2023", "打率_2024",
    "本塁打_2023", "本塁打_2024",
    "出塁率_2023", "出塁率_2024",
    "OPS_2023", "OPS_2024"
]
pivot_df = pivot_df[[col for col in cols if col in pivot_df.columns]]
pivot_df.columns = pivot_df.columns.str.replace("_(\d+)", r"（\1年）", regex=True)

st.subheader("📊 年度別成績（横並び比較）")
# ピボット後の数値データ整形
pivot_numeric = pivot_df.drop(columns=["名前"]).apply(pd.to_numeric, errors="coerce")
pivot_numeric.index = pivot_df["名前"]

# ✅ 表示フォーマット定義
format_dict = {
    col: "{:.3f}" if "打率" in col or "出塁率" in col or "OPS" in col else "{:.0f}"
    for col in pivot_numeric.columns
}

# ✅ フォーマット＋ハイライト表示
styled_df = pivot_numeric.style.format(format_dict).highlight_max(axis=1, color='lightblue')

st.dataframe(styled_df)

# 年度比較チャート
filtered_data = df[(df["名前"] == selected_player) & (df["年度"].isin(selected_years))]
st.subheader("年度別成績比較")

year_2023 = filtered_data[filtered_data["年度"] == 2023]
year_2024 = filtered_data[filtered_data["年度"] == 2024]

# ラベル（指標）
labels = ['打率', '出塁率', 'OPS']

# 数値（floatで統一）
values_2023 = [float(val) for val in year_2023[labels].values[0]]
values_2024 = [float(val) for val in year_2024[labels].values[0]]

x = np.arange(len(labels))  # 位置
width = 0.35

fig, ax = plt.subplots()
bars1 = ax.bar(x - width/2, values_2023, width, label='2023年', color='#1f77b4')
bars2 = ax.bar(x + width/2, values_2024, width, label='2024年', color='#ff7f0e')

ax.set_ylabel('成績')
ax.set_title('打撃成績比較（2023年 vs 2024年）')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# 棒の上に数値表示
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
st.pyplot(fig)

# ---------------------------
# CSV読み込み（2024年・2025年）
df_2025 = pd.read_csv("dena_2025_stats.csv")

# 前処理
for d in [df_2024, df_2025]:
    d["年度"] = pd.to_numeric(d["年度"], errors="coerce").astype("Int64")
    for col in ["打率", "OPS", "本塁打"]:
        d[col] = d[col].astype(str).str.strip()
    d["打率"] = pd.to_numeric(d["打率"], errors="coerce")
    d["OPS"] = pd.to_numeric(d["OPS"], errors="coerce")
    d["本塁打"] = pd.to_numeric(d["本塁打"], errors="coerce")

# 結合
df = pd.concat([df_2024, df_2025], ignore_index=True)

# MVPスコア
df = df.dropna(subset=["打率", "OPS", "本塁打"])
df["MVPスコア"] = df["打率"] * df["OPS"] * df["本塁打"]

# 🧲 2025年成績をスクレイピング
# 📊 Streamlit アプリ UI

st.title("横浜DeNAベイスターズ 打者成績分析（2024年 vs 2025年）")

# 関数定義
def preprocess(df):
    df["年度"] = pd.to_numeric(df["年度"], errors="coerce").astype("Int64")
    for col in ["打率", "OPS", "本塁打"]:
        df[col] = df[col].astype(str).str.strip()
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["打率", "OPS", "本塁打"])
    df["MVPスコア"] = df["打率"] * df["OPS"] * df["本塁打"]
    return df

# 使用例
df_2024 = preprocess(df_2024)
df_2025 = preprocess(df_2025)

selected_player = st.selectbox("選手を選んでください", sorted(df["名前"].unique()), key="compare_select")
player_df = df[df["名前"] == selected_player]

# 年度別データ
year_2024 = player_df[player_df["年度"] == 2024]
year_2025 = player_df[player_df["年度"] == 2025]

if not year_2024.empty and not year_2025.empty:
    labels = ['打率', '出塁率', 'OPS']
    values_2024 = [float(year_2024[label].values[0]) for label in labels]
    values_2025 = [float(year_2025[label].values[0]) for label in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    bars1 = ax.bar(x - width/2, values_2024, width, label='2024年', color='#1f77b4')
    bars2 = ax.bar(x + width/2, values_2025, width, label='2025年', color='#ff7f0e')

    ax.set_ylabel('成績')
    ax.set_title(f"{selected_player} の打撃成績比較（2024 vs 2025）")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    st.pyplot(fig)

    # 自動コメント表示
    delta_dict = {label: values_2025[i] - values_2024[i] for i, label in enumerate(labels)}
    st.markdown("### 📈 パフォーマンス変化")
    for label in labels:
        delta = delta_dict[label]
        emoji = "🔺" if delta > 0 else ("🔻" if delta < 0 else "⏺")
        st.write(f"{emoji} **{label}**：{values_2025[labels.index(label)]:.3f}（2024年との差：{delta:+.3f}）")

else:
    st.warning("選手に2024年または2025年のデータがありません。")
