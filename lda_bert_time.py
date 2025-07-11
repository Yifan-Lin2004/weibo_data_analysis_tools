# -*- coding: utf-8 -*-
import os
import re
import warnings
import multiprocessing
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pandas as pd
from tqdm import tqdm
import jieba
import torch

from gensim import corpora, models
from gensim.models import CoherenceModel
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis


# ----------------------- 全局分词函数 ----------------------- #
STOP_PATH = Path(__file__).parent / "stopword.txt"
stopwords = (
    {
        w.strip()
        for w in open(STOP_PATH, encoding="utf-8")
        if w.strip()
    }
    if STOP_PATH.exists()
    else set()
)


def clean_cut(txt: str):
    """微博文本清洗 + 分词"""
    txt = re.sub(r"#.*?#|@\w+|http\S+|O网页链接", " ", str(txt))
    txt = re.sub(r"[^\w\s\u4e00-\u9fff]", " ", txt)
    return [w for w in jieba.lcut(txt) if len(w) > 1 and w not in stopwords]


# ----------------------------- 主流程 ------------------------------------ #
def main():
    warnings.filterwarnings("ignore")
    tqdm.pandas()

    # ---------- 路径 ----------
    SCRIPT_DIR = Path(__file__).parent
    OUTPUTS_DIR = SCRIPT_DIR / "docs" / "outputs"
    MODELS_DIR = SCRIPT_DIR / "models"
    DATA_FILE = SCRIPT_DIR / "combined_annotated.csv"

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- 读取数据 ----------
    assert DATA_FILE.exists(), f"找不到数据文件: {DATA_FILE}"
    df = pd.read_csv(DATA_FILE, dtype=str)
    assert {"微博正文", "发布时间"} <= set(df.columns), "CSV 缺少必要列"

    df["发布时间"] = pd.to_datetime(df["发布时间"])

    # ---------- 仅保留 2024 年数据 ----------
    mask_2024 = (df["发布时间"] >= "2024-01-01") & (df["发布时间"] <= "2024-12-31")
    df_2024 = df.loc[mask_2024]

    if df_2024.empty:
        print("⚠️ 数据中不包含 2024 年内容，流程终止。")
        return

    segments = [("year_2024", df_2024)]

    # ---------- 嵌入模型 ----------
    embedder = SentenceTransformer(
        "paraphrase-multilingual-MiniLM-L12-v2",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # ---------- 循环处理（这里只有一个分段） ----------
    for name, sub in segments:
        texts = sub["微博正文"].fillna("").tolist()
        N = len(texts)
        print(f"\n=== 处理分段: {name}，共 {N} 条微博 ===")

        # 动态调整 vectorizer 参数
        if N < 50:
            vec = CountVectorizer(
                tokenizer=clean_cut,
                ngram_range=(1, 2),
                min_df=1,
                max_df=1.0,
            )
        else:
            vec = CountVectorizer(
                tokenizer=clean_cut,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
            )

        token_lists = [clean_cut(t) for t in texts]

        # ----- 1) BERTopic -----
        topic_model = BERTopic(
            language="chinese",
            embedding_model=embedder,
            vectorizer_model=vec,
            min_topic_size=10,
            top_n_words=12,
            calculate_probabilities=True,
            verbose=False,
        )
        topic_model.fit(texts)

        # 安全尝试自动合并主题
        try:
            topic_model = topic_model.reduce_topics(texts, nr_topics="auto")
        except Exception as e:
            print(f"⚠️ {name} auto-reduce_topics 跳过: {e}")

        seg_out = OUTPUTS_DIR / name
        seg_model = MODELS_DIR / name
        seg_out.mkdir(exist_ok=True)
        seg_model.mkdir(exist_ok=True)

        topic_model.save(str(seg_model / "bertopic_model"))
        topic_model.get_topic_info().to_csv(
            seg_out / "bertopic_topic_info.csv", index=False
        )
        topic_model.visualize_topics(width=1200, height=700).write_html(
            str(seg_out / "bertopic_topics.html")
        )

        # ----- 2) LDA (k=18) -----
        dictionary = corpora.Dictionary(token_lists)
        dictionary.filter_extremes(no_below=5, no_above=0.5)
        corpus = [dictionary.doc2bow(t) for t in token_lists]

        lda18 = models.LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=18,
            passes=10,
            random_state=42,
        )
        lda18.save(str(seg_model / "lda_18.bin"))

        with open(seg_out / "lda_top_words_18.txt", "w", encoding="utf-8") as f:
            for i, topic in lda18.show_topics(18, 15, formatted=False):
                f.write(f"Topic {i:02d}: {' '.join(w for w, _ in topic)}\n")

        vis18 = gensimvis.prepare(lda18, corpus, dictionary, sort_topics=False)
        pyLDAvis.save_html(vis18, str(seg_out / "lda_vis_18.html"))

        # ----- 3) LDA k 调优 -----
        print(" - 调参 k ...")
        scores, best_k, best_c, best_model = [], -1, -1, None
        for k in range(8, 31, 2):
            m = models.LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics=k,
                passes=10,
                random_state=42,
            )
            c = (
                CoherenceModel(
                    model=m,
                    texts=token_lists,
                    dictionary=dictionary,
                    coherence="c_v",
                ).get_coherence()
            )
            scores.append((k, c))
            if c > best_c:
                best_k, best_c, best_model = k, c, m

        ks, cvs = zip(*scores)
        plt.figure(figsize=(8, 4.5))
        plt.plot(ks, cvs, marker="o")
        plt.xlabel("Num Topics (k)")
        plt.ylabel("Coherence (c_v)")
        plt.title("LDA Coherence vs k")
        plt.grid(alpha=0.3)
        plt.xticks(ks)
        plt.tight_layout()
        plt.savefig(seg_out / "lda_k_selection.png", dpi=150)
        plt.close()

        best_model.save(str(seg_model / f"lda_best_k{best_k}.bin"))
        print(f"✅ [{name}] 最佳 k={best_k}, coherence={best_c:.4f}")

    print("\n🎉 2024 年数据处理完毕，结果已写入 docs/outputs/")


# ----------------------------- 入口 ------------------------------------- #
if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
