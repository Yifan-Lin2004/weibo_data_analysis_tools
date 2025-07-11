# 导入所需库
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib

# 设置matplotlib的中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
matplotlib.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# 读取数据（假设 CSV 文件已保存为 df）
df = pd.read_csv('combined_annotated.csv')  # 请根据实际文件路径调整
# 提取发布工具列
tools = df['发布工具'].fillna("未知")

# 识别品牌函数：直接返回中文品牌名
brand_keywords = {
    "苹果": ["iphone", "ipad", "apple","苹果"],
    "华为": ["huawei", "华为", "荣耀", "honor"],
    "小米": ["xiaomi", "小米", "mi "],
    "OPPO": ["oppo"],
    "vivo": ["vivo"],
    "三星": ["samsung", "三星"],
    "一加": ["oneplus", "一加"],
    "魅族": ["meizu", "魅族"],
    "中兴": ["zte", "中兴"],
    "联想": ["lenovo", "联想"],
    "红米": ["redmi", "红米"],
    "realme": ["realme"],
    "努比亚": ["nubia", "努比亚"],
    "酷派": ["coolpad", "酷派"],
    "金立": ["gionee", "金立"],
    "魅蓝": ["魅蓝"],
    "锤子": ["smartisan", "锤子"],
    # 可继续补充
}
def identify_brand(tool):
    tool = str(tool)
    t = tool.lower()
    for brand, keywords in brand_keywords.items():
        for kw in keywords:
            if kw in t or kw in tool:
                return brand
    return "其他安卓"

df['手机品牌_原始'] = tools.apply(identify_brand)

# 统计前五多的品牌
brand_counts_all = df['手机品牌_原始'].value_counts()
top5_brands = list(brand_counts_all.head(5).index)

def merge_top5(brand):
    if brand in top5_brands:
        return brand
    return "其他安卓"
df['手机品牌'] = df['手机品牌_原始'].apply(merge_top5)

# 只保留前五+其他安卓
brands = top5_brands + ["其他安卓"]
df['手机品牌'] = pd.Categorical(df['手机品牌'], categories=brands, ordered=True)

# 统计各品牌的帖子数
brand_counts = df['手机品牌'].value_counts().reindex(brands, fill_value=0)
result_text = "各设备/品牌发帖数量：\n"
result_text += str(brand_counts) + "\n\n"
print("各设备/品牌发帖数量：")
print(brand_counts)

# 不同品牌情感分数描述统计
brand_stats = df.groupby('手机品牌')['sent_score'].agg(['mean', 'std', 'count']).reindex(brands)
result_text += "不同品牌情感分数均值和标准差：\n"
result_text += str(brand_stats) + "\n\n"
print("\n不同品牌情感分数均值和标准差：")
print(brand_stats)

# ANOVA检验不同品牌情感均值是否有显著差异
brand_groups = [df[df['手机品牌']==b]['sent_score'].values for b in brands]
brand_groups_nonempty = [g for g in brand_groups if len(g) > 0]
anova_brand = stats.f_oneway(*brand_groups_nonempty)
result_text += f"品牌情感均值 ANOVA: F = {anova_brand.statistic:.3f}, p = {anova_brand.pvalue:.3e}\n"
print(f"\n品牌情感均值 ANOVA: F = {anova_brand.statistic:.3f}, p = {anova_brand.pvalue:.3e}")

# 保存统计结果到txt
with open("brand_sentiment_stats.txt", "w", encoding="utf-8") as f:
    f.write(result_text)

# 绘制品牌情感分数分布箱型图
plt.figure(figsize=(8,5), dpi=100)
plt.boxplot([df[df['手机品牌']==b]['sent_score'].dropna() for b in brands],
            labels=brands, showmeans=True, showfliers=False,
            meanprops={"marker":"o","markerfacecolor":"white","markeredgecolor":"black"})
plt.ylabel("情感得分")
plt.title("主流手机品牌用户的情感分数分布")
plt.ylim(-1, 1)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("brand_sentiment_box.png")
plt.close()
