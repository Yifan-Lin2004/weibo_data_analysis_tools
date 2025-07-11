# 导入所需库
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# 读取数据（假设 CSV 文件已保存为 df）
df = pd.read_csv('combined_annotated.csv')  # 请根据实际文件路径调整

# 将发布时间转换为 datetime 类型（为后续模块做准备）
df['发布时间'] = pd.to_datetime(df['发布时间'])

# 地域分类规则定义
municipalities = {"北京", "上海", "天津", "重庆"}
tier1 = {"广州", "深圳", "杭州"}
tier2 = {"成都", "南京", "武汉", "西安", "苏州", "宁波", "青岛", "长沙", "郑州", "厦门",
         "沈阳", "大连", "哈尔滨", "长春", "福州", "济南", "南宁", "合肥", "佛山"}  # 等

# 中国省级行政区名称，用于识别模糊地点（如仅省名）
provinces = {"黑龙江", "吉林", "辽宁", "内蒙古", "河北", "河南", "山东", "山西", "陕西",
             "宁夏", "甘肃", "青海", "新疆", "西藏", "四川", "云南", "贵州", "湖北",
             "湖南", "江西", "安徽", "浙江", "江苏", "福建", "广东", "广西", "海南",
             "香港", "澳门", "台湾"}

# 海外判断关键词（常见国家/城市的中英文名称片段）
overseas_keywords = ["美国", "英国", "法国", "德国", "日本", "俄罗斯", "加拿大", "澳大利亚", 
                     "新加坡", "韩国", "朝鲜", "意大利", "西班牙", "印度", "泰国", "马来西亚",
                     "Indonesia", "Tokyo", "New York", "London", "Paris", "Singapore"]

# 定义函数进行地域分类
def categorize_location(loc):
    if pd.isna(loc) or str(loc).strip() == "":
        return "未知"
    loc_str = str(loc)
    # 去除可能的定位细节，只取主地名（如 "北京·xxx"取"北京"）
    city = loc_str.split('·')[0]  
    city = city.strip()
    # 分类判断
    if city in municipalities:
        return "直辖市"
    elif city in tier1:
        return "一线城市"
    elif city in tier2:
        return "二线城市"
    # 判断海外：如果地点字符串包含海外关键字
    elif any(key.lower() in loc_str.lower() for key in overseas_keywords):
        return "海外"
    # 判断省份等模糊信息
    elif city in provinces:
        return "未知"
    else:
        # 其他未列出城市都归为三线及以下
        return "三线及以下"

# 应用地域分类
df['城市等级'] = df['发布位置'].apply(categorize_location)

# 计算各城市等级的情感均值
sentiment_means = df.groupby('城市等级')['sent_score'].mean()
result_text = "各城市等级情感均值：\n"
for level, mean_val in sentiment_means.items():
    line = f"{level}: {mean_val:.3f}"
    print(line)
    result_text += line + "\n"

# 单因素方差分析（ANOVA）显著性检验
groups = [group['sent_score'].values for _, group in df.groupby('城市等级')]
anova_result = stats.f_oneway(*groups)
result_text += f"\nANOVA 检验结果: F = {anova_result.statistic:.3f}, p = {anova_result.pvalue:.3e}\n"
print(f"\nANOVA 检验结果: F = {anova_result.statistic:.3f}, p = {anova_result.pvalue:.3e}")
if anova_result.pvalue < 0.05:
    result_text += "不同城市等级的情感均值差异具有统计显著性\n"
    print("不同城市等级的情感均值差异具有统计显著性")
else:
    result_text += "不同城市等级的情感均值差异不显著\n"
    print("不同城市等级的情感均值差异不显著")

# 保存统计结果到txt
with open("city_sentiment_stats.txt", "w", encoding="utf-8") as f:
    f.write(result_text)

# 绘制不同城市等级的情感均值柱状图（带95%置信区间）
plt.figure(figsize=(6,4), dpi=100)
order = ["直辖市","一线城市","二线城市","三线及以下","海外","未知"]
means = [sentiment_means.get(cat, 0) for cat in order]
# 计算95%置信区间误差条：1.96 * (std/√n)
sentiment_stats = df.groupby('城市等级')['sent_score'].agg(['mean','std','count'])
yerr = []
for cat in order:
    if cat in sentiment_stats.index:
        m, s, n = sentiment_stats.loc[cat]
        ci = 1.96 * s / np.sqrt(n)  # 近似95%置信区间
    else:
        ci = 0
    yerr.append(ci)
plt.bar(order, means, yerr=yerr, capsize=4, color='#6699CC')
plt.ylabel("情感均值")
plt.title("不同城市等级的平均情感分数（95%置信区间）")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("region_sentiment_bar.png")
plt.close()
