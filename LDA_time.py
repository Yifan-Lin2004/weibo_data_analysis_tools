import jieba
from gensim import corpora, models
from gensim.models import CoherenceModel
from tqdm import tqdm  # 新增
import matplotlib
import matplotlib.pyplot as plt
# import matplotlib.font_manager as fm  # 不再需要

# 导入所需库
import pandas as pd
import numpy as np
from scipy import stats

# 启用tqdm进度条用于pandas apply
from tqdm import tqdm

tqdm.pandas()

# 加载停用词表
def load_stopwords(filepath='stopword.txt'):
    with open(filepath, 'r', encoding='utf-8') as f:
        return set([line.strip() for line in f if line.strip()])

# 文本预处理函数，支持外部停用词
def preprocess_text(text, stopwords=None):
    import re
    text = re.sub(r"#.*?#", " ", str(text))        # 去除形如#话题#
    text = re.sub(r"@[\w\-\u4e00-\u9fff]+", " ", text)  # 去除@用户
    text = re.sub(r"http[s]?://\S+|O网页链接", " ", text)  # 去除URL
    text = re.sub(r"[^\w\s\u4e00-\u9fff]", " ", text)
    words = jieba.lcut(text)
    if stopwords is None:
        stopwords = set()
    words = [w for w in words if w.strip() and w not in stopwords and len(w) > 1]
    return words

def main():
    # 设置matplotlib中文字体（推荐方式，避免找不到本地字体文件报错）
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
    matplotlib.rcParams['axes.unicode_minus'] = False    # 正常显示负号
    # 读取数据（假设 CSV 文件已保存为 df）
    df = pd.read_csv('combined_annotated.csv')  # 请根据实际文件路径调整

    # 1. 加载停用词
    stopwords = load_stopwords('stopword.txt')
    # 2. 分词时传入停用词
    texts = df['微博正文'].progress_apply(lambda x: preprocess_text(x, stopwords))

    # 构建词典和语料
    dictionary = corpora.Dictionary(texts)
    # 过滤极端词：出现少于5次或过于普遍(>50%文档)的词
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # 确定最佳主题数 (尝试5~10主题，根据Coherence值)
    coherence_scores = []
    best_model = None
    best_k = 0
    best_score = -1
    # 用tqdm包裹for循环
    for k in tqdm(range(5, 11), desc="LDA主题数遍历"):
        lda_model = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=k, 
                                    random_state=42, passes=10)
        # 计算一致性得分
        cm = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
        score = cm.get_coherence()
        coherence_scores.append((k, score))
        print(f"{k} 个主题的一致性得分: {score:.4f}")
        if score > best_score:
            best_score = score
            best_k = k
            best_model = lda_model

    print(f"\n选择的最佳主题数为: {best_k}, 对应一致性得分: {best_score:.4f}")

    # 输出每个主题的关键词
    topics_keywords = {}
    for topic_id in range(best_k):
        top_terms = [word for word, prob in best_model.show_topic(topic_id, topn=5)]
        topics_keywords[topic_id] = "、".join(top_terms)
        print(f"主题{topic_id+1}: {topics_keywords[topic_id]}")

    # 每篇微博分配主主题（概率最高的主题）
    doc_topics = [max(best_model.get_document_topics(bow), key=lambda x: x[1])[0] if bow else None 
                  for bow in corpus]
    df['主题'] = doc_topics

    # 按月份统计每月各主题频数
    df['月份'] = pd.to_datetime(df['发布时间']).dt.to_period('M')
    topic_trend = df.groupby(['月份','主题']).size().unstack(fill_value=0)

    # 将索引转换为时间序列并按时间排序
    topic_trend = topic_trend.reindex(pd.period_range(df['月份'].min(), df['月份'].max(), freq='M'), fill_value=0)
    topic_trend.index = topic_trend.index.to_timestamp()  # 转为Timestamp索引

    # 绘制主题演化堆积面积图
    plt.figure(figsize=(8,5), dpi=100)
    # 按每列主题的频次作面积图
    labels = [f"主题{tid+1}: "+topics_keywords.get(tid, "") for tid in topic_trend.columns]
    plt.stackplot(topic_trend.index, topic_trend.T, labels=labels, alpha=0.8)
    plt.xlabel("时间")
    plt.ylabel("帖数")
    plt.title("微博主要主题随时间的趋势（按月）")
    plt.legend(loc='upper left', bbox_to_anchor=(1.05,1.0))
    plt.tight_layout()
    plt.savefig("topic_trends_area.png")
    plt.close()

if __name__ == "__main__":
    main()
