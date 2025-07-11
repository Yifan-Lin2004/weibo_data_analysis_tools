
import os, pandas as pd, numpy as np, networkx as nx, seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from community import community_louvain
from datetime import datetime

plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False    
# 自动创建输出目录

os.makedirs("./figs", exist_ok=True)
DATA_PATH = "./combined_annotated2.csv"    # 修改为你的 CSV
dtype_map = {"id": str, "retweet_id": str, "user_id": str}
df = pd.read_csv(DATA_PATH, dtype=dtype_map)

def clean(col):
    return col.fillna("").astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
df["id"], df["retweet_id"], df["user_id"] = map(clean, [df["id"], df["retweet_id"], df["user_id"]])

df_retweet = df[df["retweet_id"] != ""]
print(f"★ 转发微博数量：{len(df_retweet):,}")

###############################################################
# 1. 构建“转发”网络 —— 节点=用户，边=谁转发了谁
###############################################################
orig_map = df[["id", "user_id"]].rename(columns={"id": "retweet_id", "user_id": "orig_user"})
edges_df = (df_retweet[["user_id", "retweet_id"]]
            .merge(orig_map, on="retweet_id", how="left")
            .dropna(subset=["orig_user"]))

print(f"★ 成功映射“转发→原作者”关系：{len(edges_df):,} 条")

G = nx.DiGraph()
G.add_nodes_from(pd.concat([edges_df["user_id"], edges_df["orig_user"]]).unique())
G.add_edges_from(edges_df[["user_id", "orig_user"]].itertuples(index=False, name=None))
print(f"★ 网络规模：{G.number_of_nodes()} 个用户节点, {G.number_of_edges()} 条转发边")

# — 导出 Gephi (可选)
nx.write_gexf(G, "./retweet_network.gexf")

###############################################################
# 2. 网络全局指标
###############################################################
def avg_out_deg(g): return np.mean([d for _, d in g.out_degree()])
N, E, avg_deg, dens = G.number_of_nodes(), G.number_of_edges(), avg_out_deg(G), nx.density(G)
largest_cc = max(nx.weakly_connected_components(G), key=len)
G_cc = G.subgraph(largest_cc).to_undirected()
avg_clust = nx.average_clustering(G_cc) if G_cc.number_of_edges() else np.nan
avg_path  = nx.average_shortest_path_length(G_cc) if G_cc.number_of_edges() else np.nan
diameter  = nx.diameter(G_cc) if G_cc.number_of_edges() else np.nan

print("\n★ 全局拓扑指标")
print(pd.DataFrame({"节点数":[N],"边数":[E],"平均度":[avg_deg],
                    "平均聚类系数":[avg_clust],"平均路径长度":[avg_path],
                    "网络直径":[diameter],"网络密度":[dens]}).to_string(index=False))

###############################################################
# 3. 网络可视化（图 6-L/R）
###############################################################
def draw_network(sample_G, title, with_colorbar=False):
    plt.figure(figsize=(9, 9))
    pos = nx.spring_layout(sample_G, k=0.22, seed=42)

    indeg = np.array([d for _, d in sample_G.in_degree()])
    # 节点大小：小节点更小，大节点更大
    min_size = 10
    max_size = 2000
    if indeg.max() > 0:
        sizes = min_size + (max_size - min_size) * ((indeg - indeg.min()) / (indeg.max() - indeg.min())) ** 2.2
    else:
        sizes = np.full_like(indeg, min_size)

    nodes = nx.draw_networkx_nodes(
        sample_G, pos,
        node_size=sizes,
        node_color=indeg,
        cmap="plasma",  # 更现代的色系
        alpha=0.95,
        linewidths=1.2,
        edgecolors="#222222"
    )
    nx.draw_networkx_edges(
        sample_G, pos,
        width=1.2,
        alpha=0.45,
        edge_color="#AAB7B8"
    )
    plt.title(title, fontsize=20, fontweight='bold', pad=18)
    plt.axis("off")
    plt.tight_layout(pad=1.5)

# 6-L：抽样 3k 边
sample_edges = list(G.edges())[:3000]
draw_network(nx.DiGraph(sample_edges), "转发网络（抽样）")
plt.tight_layout(); plt.savefig("./figs/转发网络.png"); plt.show()

# 6-R：移除最高入度节点
top_node = max(G.in_degree, key=lambda x: x[1])[0]
G_removed = G.copy(); G_removed.remove_node(top_node)
draw_network(nx.DiGraph(list(G_removed.edges())[:3000]), "移除中心节点后")
plt.tight_layout(); plt.savefig("./figs/去中心.png"); plt.show()

###############################################################
# 4. 度分布（图 4 & 7）
###############################################################
indeg_vals = [d for _, d in G.in_degree()]
# 图 7：直方图
plt.figure(figsize=(7,5))
sns.histplot(indeg_vals, bins=25, log_scale=(False,True),
             color=sns.color_palette("rocket")[1], edgecolor="k", linewidth=.3)
plt.xlabel("入度（被转发次数）"); plt.ylabel("节点数（对数刻度）")
plt.title("入度分布", pad=12)
plt.tight_layout(); plt.savefig("./figs/入度分布.png"); plt.show()
# 图 4：双对数散点
cnt = Counter(indeg_vals); x,y=zip(*cnt.items())
plt.figure(figsize=(5.5,4.5)); plt.scatter(x,y,s=30,alpha=.75,
                                           color=sns.color_palette("rocket")[0])
plt.xscale("log"); plt.yscale("log")
plt.xlabel("入度 (log)"); plt.ylabel("频次 (log)")
plt.title("度分布（双对数）", pad=10)
plt.tight_layout(); plt.savefig("./figs/双对数.png"); plt.show()

###############################################################
# 5. 社区发现 & 可视化（图 8/9）
###############################################################
partition = community_louvain.best_partition(G.to_undirected())
G_sample = nx.DiGraph(sample_edges)               # 重用抽样子图
colors = [partition[n] for n in G_sample.nodes()]
cmap = sns.color_palette("tab20", len(set(colors)))
plt.figure(figsize=(9, 9))
pos_sample = nx.spring_layout(G_sample, k=0.22, seed=42)
indeg_sample = np.array([d for _, d in G_sample.in_degree()])
min_size = 10
max_size = 500
if indeg_sample.max() > 0:
    sizes_sample = min_size + (max_size - min_size) * ((indeg_sample - indeg_sample.min()) / (indeg_sample.max() - indeg_sample.min())) ** 2.2
else:
    sizes_sample = np.full_like(indeg_sample, min_size)
nx.draw_networkx_nodes(
    G_sample, pos_sample,
    node_size=sizes_sample,
    node_color=indeg_sample,
    cmap="plasma",
    alpha=0.95,
    linewidths=1.2,
    edgecolors="#222222"
)
nx.draw_networkx_edges(
    G_sample, pos_sample,
    width=1.2,
    alpha=0.45,
    edge_color="#AAB7B8"
)
plt.title("Figure 8/9  |  Louvain 社区划分", fontsize=20, fontweight='bold', pad=18)
plt.axis("off"); plt.tight_layout(pad=1.5)
plt.savefig("./figs/Figure8_communities.png"); plt.show()

###############################################################
# 6. 最长转发链示例（可选打印）
###############################################################
weibo2user = dict(zip(df["id"], df["user_id"]))
def longest_chain(df_rt):
    chains=[]
    for _,r in df_rt.iterrows():
        chain=[r["user_id"]]; tgt=weibo2user.get(r["retweet_id"])
        while tgt and tgt!=chain[-1]:
            chain.append(tgt); break     # 仅示意：最多追溯一级
        chains.append(chain)
    return max(chains,key=len)
lc = longest_chain(df_retweet)
print("★ 观察到最长转发链（用户 id 顺序）:", " ➝ ".join(lc))

###############################################################
# 7. 关键节点测度（图 11-17）
###############################################################
# —— K-Shell
G_und = G.to_undirected(); G_und.remove_edges_from(nx.selfloop_edges(G_und))
core = nx.core_number(G_und); max_k = max(core.values())
shell_cnt = Counter(core.values())
plt.figure(figsize=(6,4.5))
sns.barplot(x=list(shell_cnt.keys()), y=list(shell_cnt.values()), palette="rocket_r")
plt.xlabel("K-Shell 层"); plt.ylabel("节点数"); plt.title("K-Shell 分布", pad=10)
plt.tight_layout(); plt.savefig("./figs/KShell.png"); plt.show()
draw_network(G.subgraph([n for n,k in core.items() if k==max_k]), f"{max_k}-壳核心子图")
plt.tight_layout(); plt.savefig("./figs/KShell核心.png"); plt.show()

# —— 中心性 bar 图
# 新bar函数，支持自定义y轴label
def bar_with_label(data, title, file, labels):
    def short_label(label, maxlen=20, mask_last=False):
        if not isinstance(label, str):
            label = str(label)
        if mask_last and len(label) > 0:
            label = label[:-1] + '*'
        return label[:maxlen] + ('...' if len(label) > maxlen else '')
    nodes, vals = zip(*data)
    # 判断是否为昵称映射
    is_nickname = labels is user_id_to_nickname
    x_labels = [short_label(labels.get(n, str(n)), maxlen=20, mask_last=is_nickname) for n in nodes]
    plt.figure(figsize=(8,5))
    sns.barplot(x=x_labels, y=vals, orient="v", palette="rocket_r")
    plt.xticks(rotation=30, ha='right', fontsize=10)
    plt.title(title, pad=10); plt.ylabel("数量"); plt.xlabel("")
    sns.despine(bottom=True)
    plt.tight_layout(); plt.savefig(file); plt.show()

# 构建id到昵称、正文的映射
user_id_to_nickname = dict(zip(df["user_id"], df["用户昵称"]))
user_id_to_text = dict(zip(df["user_id"], df["微博正文"]))

# 度中心性 Top10
bar_with_label(
    sorted(G.degree, key=lambda x:x[1], reverse=True)[:10],
    "度中心性 Top10（昵称）", "./figs/度中心性_昵称.png", user_id_to_nickname
)
bar_with_label(
    sorted(G.degree, key=lambda x:x[1], reverse=True)[:10],
    "度中心性 Top10（正文摘要）", "./figs/度中心性_正文.png", user_id_to_text
)
# 入度 Top10
bar_with_label(
    sorted(G.in_degree, key=lambda x:x[1], reverse=True)[:10],
    "入度 Top10（昵称）", "./figs/入度_昵称.png", user_id_to_nickname
)
bar_with_label(
    sorted(G.in_degree, key=lambda x:x[1], reverse=True)[:10],
    "入度 Top10（正文摘要）", "./figs/入度_正文.png", user_id_to_text
)
# 出度 Top10
bar_with_label(
    sorted(G.out_degree, key=lambda x:x[1], reverse=True)[:10],
    "出度 Top10（昵称）", "./figs/出度_昵称.png", user_id_to_nickname
)
bar_with_label(
    sorted(G.out_degree, key=lambda x:x[1], reverse=True)[:10],
    "出度 Top10（正文摘要）", "./figs/出度_正文.png", user_id_to_text
)

closeness  = nx.closeness_centrality(G_cc)
constraint = nx.constraint(G_cc)
eigen      = nx.eigenvector_centrality(G_cc, max_iter=1000)

# 图 15：Closeness >0.5
high_close = {n: v for n, v in closeness.items() if v > 0.5}
s = pd.Series(high_close).sort_values(ascending=False)
if not s.empty:
    s.plot(kind="barh", color=sns.color_palette("rocket")[2])
    plt.title("Figure 15  |  Closeness > 0.5", pad=8)
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig("./figs/Figure15_closeness.png")
    plt.show()
else:
    print('没有closeness>0.5的节点，跳过该图绘制')

# 图 16：Constraint vs Closeness
plt.figure(figsize=(5.5,4.5))
plt.scatter(list(constraint.values()), list(closeness.values()),
            s=24, alpha=.75, color=sns.color_palette("rocket")[1])
plt.xlabel("限制度（越小结构洞越明显）"); plt.ylabel("接近中心性")
plt.title("结构洞限制度 vs 接近中心性", pad=10)
plt.tight_layout(); plt.savefig("./figs/限制度_vs_接近.png"); plt.show()

# 图 17：Eigenvector
plt.figure(figsize=(5.5,3))
plt.scatter(list(eigen.values()), [1]*len(eigen), s=28, alpha=.6,
            color=sns.color_palette("rocket")[0])
plt.xlabel("特征向量中心性"); plt.yticks([])
plt.title("特征向量中心性分布", pad=8)
plt.tight_layout(); plt.savefig("./figs/特征向量中心性.png"); plt.show()

###############################################################
# 8. 意见领袖识别 + 指标导出
###############################################################
thr_in    = np.percentile([d for _, d in G.in_degree()], 99)
thr_close = np.percentile(list(closeness.values()), 99)
leaders=[n for n in G_cc.nodes() if G.in_degree(n)>=thr_in and core[n]==max_k
         and closeness[n]>=thr_close and constraint.get(n,1)<0.25]
print(f"★ 潜在意见领袖 {len(leaders)} 人：", leaders)

pd.DataFrame({
    "user_id": list(G_cc.nodes()),
    "in_degree": [G.in_degree(n) for n in G_cc],
    "out_degree":[G.out_degree(n) for n in G_cc],
    "k_shell":   [core[n] for n in G_cc],
    "closeness": [closeness[n] for n in G_cc],
    "constraint":[constraint.get(n,np.nan) for n in G_cc],
    "eigen":     [eigen[n] for n in G_cc]
}).to_csv("./retweet_node_metrics.csv", index=False)
print("★ 指标已导出：./retweet_node_metrics.csv")
