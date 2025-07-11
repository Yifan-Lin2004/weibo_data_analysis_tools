import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  
matplotlib.rcParams['axes.unicode_minus'] = False    
df = pd.read_csv('combined_annotated.csv') 

# 区分会员(认证)类型：普通用户、黄V(含金V红V)、蓝V
def categorize_user_type(auth):
    if auth == "普通用户":
        return "普通用户"
    elif "蓝V" in auth:
        return "蓝V"
    else:
        # 将黄V、金V、红V等个人认证都归为黄V一类
        return "黄V"

df['用户类型'] = df['user_authentication'].apply(categorize_user_type)

# 增加是否大V标识：蓝V或黄V类别为大V
df['is_bigv'] = df['用户类型'].apply(lambda x: True if x in ["黄V","蓝V"] else False)

# 统计各用户类型的情感均值和数量
user_stats = df.groupby('用户类型')['sent_score'].agg(['mean','std','count'])
print("不同用户类型情感均值：\n", user_stats)

# 比较普通用户 vs 大V 的情感差异（t检验）
normal_scores = df[df['用户类型']=="普通用户"]['sent_score']
bigv_scores = df[df['is_bigv']==True]['sent_score']
ttest = stats.ttest_ind(normal_scores, bigv_scores, equal_var=False)
print(f"\n普通用户 vs 大V情感得分 t检验: t = {ttest.statistic:.3f}, p = {ttest.pvalue:.3f}")

# 比较三类用户(普通、蓝V、黄V)情感均值差异（ANOVA）
groups_user = [group['sent_score'].values for _, group in df.groupby('用户类型')]
anova_user = stats.f_oneway(*groups_user)
print(f"普通/蓝V/黄V 三组 ANOVA: F = {anova_user.statistic:.3f}, p = {anova_user.pvalue:.3e}")

# 绘制不同用户类型的情感均值柱状图
plt.figure(figsize=(5,4), dpi=100)
types = ["普通用户","黄V","蓝V"]
means = [user_stats.loc[t]['mean'] for t in types]
# 计算95%置信区间误差条
yerrs = []
for t in types:
    m, s, n = user_stats.loc[t]
    ci = 1.96 * s / np.sqrt(n)
    yerrs.append(ci)
plt.bar(types, means, yerr=yerrs, color=["#66C2A5","#FFA726","#42A5F5"], capsize=4)
plt.ylabel("平均情感分数")
plt.title("不同认证类型用户的平均情感分数")
plt.tight_layout()
plt.savefig("user_type_sentiment_bar.png")
plt.close()

with open("user_type_sentiment_stats.txt", "w", encoding="utf-8") as f:
    f.write("不同用户类型情感均值：\n")
    f.write(str(user_stats))
    f.write("\n\n")
    f.write(f"普通用户 vs 大V情感得分 t检验: t = {ttest.statistic:.3f}, p = {ttest.pvalue:.3e}\n")
    f.write(f"普通/蓝V/黄V 三组 ANOVA: F = {anova_user.statistic:.3f}, p = {anova_user.pvalue:.3e}\n")
