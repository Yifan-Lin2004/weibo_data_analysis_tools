import matplotlib as mpl
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': [
        'Microsoft YaHei', 'SimHei', 'STHeiti', 'Heiti TC',
        'PingFang SC', 'WenQuanYi Micro Hei'
    ],
    'axes.unicode_minus': False
})

import pandas as pd, seaborn as sns, matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats
import re, warnings
warnings.filterwarnings('ignore')

# ---------- 0. 列名设置 ----------
RAW_CSV   = 'combined_annotated.csv'   # 已含 sent_score
SCORE_COL = 'sent_score'               # 情感得分列
TOOL_COL  = '发布工具'                  # 手机型号原列
LOC_COL   = 'ip_location' if 'ip_location' in pd.read_csv(RAW_CSV, nrows=1).columns else '发布位置'
MEMBER_COL = '会员类型'                 # 三类：普通用户 / 蓝v / 黄v
# ---------------------------------

print('🛠  读取数据…')
df = pd.read_csv(RAW_CSV)

# ---------- 1. 手机品牌 ----------
def get_brand(text: str) -> str:
    if pd.isna(text): return '未知'
    brands = ['iPhone', 'HUAWEI', 'HONOR', 'Xiaomi', 'OPPO',
              'vivo', 'Samsung', 'OnePlus', 'Redmi', 'meizu']
    for b in brands:
        if b.lower() in text.lower():
            return b
    return '其他'
df['phone_brand'] = df[TOOL_COL].astype(str).apply(get_brand)

# ---------- 2. 地域 ----------
def get_region(text: str) -> str:
    if pd.isna(text): return '未知'
    first = re.split(r'[ \-]', text.strip())[0]
    if first in ['海外', '其他', ''] or (len(first) > 4 and first.isalpha()):
        return '海外/其他'
    return first
df['region'] = df[LOC_COL].astype(str).apply(get_region)

# ---------- 3. 会员类型 & 大V 二分类 ----------
df['member_type'] = df[MEMBER_COL].fillna('普通用户')
df['is_bigv'] = df['member_type'].apply(
    lambda x: '大V' if x in ['蓝v', '黄v', '蓝V', '黄V'] else '普通用户'
)

# ---------- 4. 保存中间表 ----------
out_cols = [SCORE_COL, 'phone_brand', 'region', 'member_type', 'is_bigv']
df[out_cols].to_csv('phone_region_bigv.csv', index=False, encoding='utf-8-sig')
print('✅ 中间数据已保存为 phone_region_bigv.csv')

# ---------- 5. 描述性可视化 ----------
sns.set_style('whitegrid')

# 5.1 品牌
top_brands = df['phone_brand'].value_counts().head(8).index
plt.figure(figsize=(10,5))
sns.violinplot(x='phone_brand', y=SCORE_COL,
               data=df[df['phone_brand'].isin(top_brands)],
               palette='Set2')
plt.axhline(0, ls='--', c='gray'); plt.title('不同手机品牌情感分布')
plt.xlabel('手机品牌'); plt.ylabel('情感得分 (-1~1)')
plt.tight_layout(); plt.savefig('brand_violin.png', dpi=300)

# 5.2 地域
top_regions = df['region'].value_counts().head(10).index
plt.figure(figsize=(10,5))
sns.boxplot(x='region', y=SCORE_COL,
            data=df[df['region'].isin(top_regions)],
            palette='Set3')
plt.axhline(0, ls='--', c='gray'); plt.title('不同地域情感分布')
plt.xlabel('地域'); plt.ylabel('情感得分 (-1~1)')
plt.xticks(rotation=45); plt.tight_layout()
plt.savefig('region_box.png', dpi=300)

# 5.3 会员类型
plt.figure(figsize=(6,5))
sns.boxplot(x='member_type', y=SCORE_COL, data=df, palette='Pastel1')
plt.axhline(0, ls='--', c='gray'); plt.title('会员类型情感分布')
plt.xlabel('会员类型'); plt.ylabel('情感得分 (-1~1)')
plt.tight_layout(); plt.savefig('member_box.png', dpi=300)

print('📊 图表已生成：brand_violin.png, region_box.png, member_box.png')

# ---------- 6. 统计检验 ----------
report = []

# 6.1 ANOVA：品牌
anova_brand = sm.stats.anova_lm(
    ols(f'{SCORE_COL} ~ C(phone_brand)', data=df).fit(), typ=2)
report.append('=== One-way ANOVA: phone_brand ===\n')
report.append(anova_brand.to_string() + '\n\n')

# 6.2 ANOVA：地域
anova_region = sm.stats.anova_lm(
    ols(f'{SCORE_COL} ~ C(region)', data=df).fit(), typ=2)
report.append('=== One-way ANOVA: region ===\n')
report.append(anova_region.to_string() + '\n\n')

# 6.3 ANOVA：会员类型（三组）
anova_member = sm.stats.anova_lm(
    ols(f'{SCORE_COL} ~ C(member_type)', data=df).fit(), typ=2)
report.append('=== One-way ANOVA: member_type ===\n')
report.append(anova_member.to_string() + '\n\n')

# 6.4 二分类 t 检验：大V vs 普通
bigv_scores   = df[df['is_bigv']=='大V'][SCORE_COL]
norm_scores   = df[df['is_bigv']=='普通用户'][SCORE_COL]
t_stat, p_val = stats.ttest_ind(bigv_scores, norm_scores, equal_var=False)
report.append('=== Two-sample t-test: 大V vs 普通用户 ===\n')
report.append(f't = {t_stat:.4f},  p = {p_val:.4g}\n\n')

# 6.5 OLS 回归：品牌 + 地域 + 大V
lm_full = ols(f'{SCORE_COL} ~ C(phone_brand) + C(region) + C(is_bigv)', data=df).fit()
report.append('=== OLS: sent_score ~ phone_brand + region + is_bigv ===\n')
report.append(lm_full.summary().as_text())

with open('stats_report.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(report))

print('📄  统计结果已写入 stats_report.txt')
print('🎉  全流程完成！')
