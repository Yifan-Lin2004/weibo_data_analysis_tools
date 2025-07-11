import os, pandas as pd, matplotlib.pyplot as plt, matplotlib as mpl, seaborn as sns
CSV      = "combined_annotated.csv"      # 含“发布时间”列
OUT_DIR  = "outputs"; os.makedirs(OUT_DIR, exist_ok=True)
PNG_PATH = os.path.join(OUT_DIR, "weibo_timeline.png")
import matplotlib as mpl
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = [
    'Microsoft YaHei', 'SimHei', 'STHeiti', 'Heiti TC',
    'PingFang SC', 'WenQuanYi Micro Hei'
]
mpl.rcParams['axes.unicode_minus'] = False
# ---------- 1. 读取 & 预处理 ----------
df = pd.read_csv(CSV, dtype=str)
df['发布时间'] = pd.to_datetime(df['发布时间'], errors='coerce')
df = df.dropna(subset=['发布时间'])

# 日期范围可自行截断
start, end = pd.to_datetime("2020-03-01"), pd.to_datetime("2025-07-04")
mask = (df['发布时间'] >= start) & (df['发布时间'] <= end)
df = df[mask]

# ---------- 2. 按日计数 + 7日移动平均 ----------
daily = (df.set_index('发布时间')
           .resample('D')
           .size()
           .rename('count'))
ma7   = daily.rolling(window=7, center=True).mean()

# ---------- 3. 关键事件 ----------
events = { 
    '生成式AI管理办法(草案)'              : '2023-04-11',
    '浙江AI陪诊师上线'                   :'2024-4-2',
    'AI宠物医生在上海展出'               :'2024-9-6',
    'DeepSeek爆火'                       : '2025-02-10',
    '“AI延误就医”舆情'                   : '2025-03-31',
    '支付宝推出AI帮看病APP'               : '2025-06-26'
}
events = {k: pd.to_datetime(v) for k,v in events.items()}

# ---------- 4. 绘图 ----------
plt.figure(figsize=(11,5))
plt.plot(daily.index.to_numpy(), daily.to_numpy(), color='#9EC9E2', label='每日数量', alpha=.6)
plt.plot(ma7.index.to_numpy(),  ma7.to_numpy(),  color='#1565C0', label='7日移动平均', linewidth=1.8)

# 事件注释
y_max = daily.max()
for name, date in events.items():
    if start <= date <= end:
        plt.axvline(date, color='grey', ls='--', lw=.8, alpha=.7)
        plt.text(date, y_max*0.92, name, rotation=90,
                 va='top', ha='right', fontsize=9)

plt.title("微博数量随时间变化（含关键事件）")
plt.xlabel("日期"); plt.ylabel("条数")
plt.legend()
plt.tight_layout()
plt.savefig(PNG_PATH, dpi=300)
plt.close()
print("✅ 时间趋势图已保存:", PNG_PATH)
