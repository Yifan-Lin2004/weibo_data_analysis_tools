import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr, kruskal
import seaborn as sns
from pathlib import Path
from matplotlib import font_manager
import matplotlib
matplotlib.use('Agg')

# ============ 0) 全局配置 ============ #
plt.rcParams["font.family"] = "SimHei"  # 如果仍乱码，把 simhei.ttf 放到脚本目录
plt.rcParams["axes.unicode_minus"] = False
OUT_DIR = Path("fig_output")
OUT_DIR.mkdir(exist_ok=True)

# ============ 1) 读取数据 ============ #
DATA_FILE = "combined_annotated.csv"      
df = pd.read_csv(DATA_FILE)

# 保留必需列 & 去空值
df = df[["id", "点赞数", "评论数", "转发数", "sent_score", "话题"]]
df = df.dropna()

# ============ 2) 拆分多话题、清洗、同义词合并 ============ #
topic_raw_col = "话题"

df[topic_raw_col] = (
    df[topic_raw_col]
      .astype(str)
      .str.replace(r"[#【】\[\]]", "", regex=True)           # 去井号/书名号
      .str.replace("，|,|/|；|;|\\|\|", " ", regex=True)     # 统一分隔符为空格
      .str.strip()
)
df = (df
      .assign(topic_list=df[topic_raw_col].str.split())
      .explode("topic_list")
      .rename(columns={"topic_list": "topic"}))

df["topic"] = df["topic"].str.lower().str.strip()

# 同义词映射（按需扩充）
syn_map = {
    "ai问诊": "ai医疗",
    "人工智能问诊": "ai医疗",
    "智能诊疗": "ai医疗",
    "covid-19": "疫情",
    "新冠疫情": "疫情",
}
df["topic_clean"] = df["topic"].replace(syn_map)

# 仅保留出现 ≥ k 次的热门话题
k = 20
hot_topics = df["topic_clean"].value_counts()
df = df[df["topic_clean"].isin(hot_topics[hot_topics >= k].index)]

# 只保留与ai、deepseek等相关的话题
keywords = ["ai", "deepseek","人工智能"]
df = df[df["topic_clean"].str.contains("|".join(keywords), case=False, na=False)]

# ============ 3) 互动量预处理 ============ #
for col in ["点赞数", "评论数", "转发数"]:
    df[col] = df[col].clip(lower=0)
    df[f"log_{col}"] = np.log1p(df[col])  # log1p 便于可视化

# ============ 4) 生成话题 numeric id ============ #
df["topic_id"] = pd.factorize(df["topic_clean"])[0]

# ============ 5) 情感分箱 ============ #
df["sent_bin"] = pd.cut(
    df["sent_score"], bins=[0, 0.33, 0.66, 1],
    labels=["negative", "neutral", "positive"]
)

# ============ 6) 相关性分析 ============ #
def corr(metric: str):
    sp_r, sp_p = spearmanr(df["sent_score"], df[metric])
    pe_r, pe_p = pearsonr(df["sent_score"], df[metric])
    return metric, sp_r, sp_p, pe_r, pe_p

corr_res = pd.DataFrame(
    [corr(m) for m in ["点赞数", "评论数", "转发数"]],
    columns=["metric", "spearman_r", "spearman_p", "pearson_r", "pearson_p"]
)
print("情感分数 × 互动量相关性：\n", corr_res.round(3), "\n")

# ============ 7) 散点图（情感 vs. log 互动量） ============ #
for m in ["点赞数", "评论数", "转发数"]:
    plt.figure(figsize=(5, 4))
    plt.scatter(df["sent_score"], df[f"log_{m}"], s=8, alpha=0.3)
    plt.xlabel("sent_score score (0–1)")
    plt.ylabel(f"log1p({m})")
    plt.title(f"sent_score vs. {m}")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"scatter_sent_score_{m}.png", dpi=300)
    plt.close()

# ============ 8) 情感分箱 × 互动量箱线图 ============ #
for m in ["点赞数", "评论数", "转发数"]:
    plt.figure(figsize=(5, 4))
    df.boxplot(column=f"log_{m}", by="sent_bin", grid=False)
    plt.suptitle("")
    plt.title(f"{m.capitalize()} distribution per sent_score bin")
    plt.xlabel("sent_score category")
    plt.ylabel(f"log1p({m})")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"box_sentbin_{m}.png", dpi=300)
    plt.close()

# ============ 9) Kruskal-Wallis：不同话题互动量差异 ============ #
kw_stats = []
for m in ["点赞数", "评论数", "转发数"]:
    grp_vals = [grp[m].values for _, grp in df.groupby("topic_id")]
    H, p = kruskal(*grp_vals)
    kw_stats.append({"metric": m, "H": H, "p": p})
print("Kruskal-Wallis 结果：\n", pd.DataFrame(kw_stats).round(3), "\n")

# ============ 10) 话题平均互动量条形图 ============ #
topic_means = (
    df.groupby("topic_clean")[["点赞数", "评论数", "转发数"]]
      .mean()
      .sort_values("点赞数", ascending=False)
)

for m, m_cn in zip(["点赞数", "评论数", "转发数"], ["点赞数", "评论数", "转发数"]):
    # 只保留均值大于0的话题，并取前20个
    topic_means_nonzero = topic_means[topic_means[m] > 0]
    top20 = topic_means_nonzero.sort_values(m, ascending=False).head(20)
    plt.figure(figsize=(16, 8))
    top20[m].plot(kind="bar")
    plt.ylabel(f"{m_cn}均值", fontsize=14)
    plt.xlabel("话题", fontsize=14)
    plt.title(f"前20话题的{m_cn}均值（每个话题≥{k}条）", fontsize=16)
    plt.xticks(rotation=60, ha="right", fontsize=11)
    plt.yticks(fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"bar_topic_{m}.png", dpi=300)
    plt.close()

# ============ 11) 话题 × 情感热力图（以点赞为例） ============ #
pivot = (df.groupby(["topic_clean", "sent_bin"])["点赞数"]
           .mean()
           .unstack())

# 选取点赞数均值最高的前20个话题
top20_topics = (
    df.groupby("topic_clean")["点赞数"]
      .mean()
      .sort_values(ascending=False)
      .head(20)
      .index
)

# 只保留这20个话题的数据
top20_pivot = pivot.loc[top20_topics]

plt.figure(figsize=(12, 10))  # 增大画布
sns.heatmap(
    top20_pivot,
    annot=True,
    fmt=".1f",
    cmap="YlGnBu",
    annot_kws={"fontsize":8}  # 缩小字体
)
plt.title("Mean 点赞数 by Top 20 Topic × sent_score Bin")
plt.tight_layout()
plt.savefig(OUT_DIR / "heatmap_topic_sent_score_点赞数.png", dpi=300)
plt.close()

print(f"全部图表已保存至文件夹：{OUT_DIR.resolve()}")
