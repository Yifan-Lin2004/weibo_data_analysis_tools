import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = [
    'Microsoft YaHei', 'SimHei', 'STHeiti', 'Heiti TC',
    'PingFang SC', 'WenQuanYi Micro Hei'
]
mpl.rcParams['axes.unicode_minus'] = False
CSV = 'combined_annotated1.csv'
df  = pd.read_csv(CSV)
df['发布时间'] = pd.to_datetime(df['发布时间'], errors='coerce')

# 若想改阈值重新分类，只需在此重新 map；否则沿用 sent_score
weekly_idx = (df.groupby(pd.Grouper(key='发布时间', freq='W'))['sent_score']
                .mean().fillna(0).sort_index())
weekly_smooth = weekly_idx.rolling(4, center=True, min_periods=1).mean()

plt.figure(figsize=(12,6))
plt.plot(weekly_idx.index.to_numpy(), weekly_idx.values, color='steelblue', alpha=.35,
         label='净情感指数（周均值）')
plt.plot(weekly_smooth.index.to_numpy(), weekly_smooth.values, color='steelblue', lw=2,
         label='4 周移动平均')
plt.axhline(0, ls='--', c='gray')

events = { 
    '浙江AI陪诊师上线'                   :'2024-4-2',
    'AI宠物医生在上海展出'               :'2024-9-6',
    'DeepSeek爆火'                       : '2025-02-10',
    '“AI延误就医”舆情'                   : '2025-03-31',
    '支付宝推出AI帮看病APP'               : '2025-06-26'
}
for name, d in events.items():
    x = pd.to_datetime(d)
    plt.axvline(x, color='gray', ls='--', alpha=.5)
    plt.text(x, plt.ylim()[1]*.95, name, rotation=90,
             ha='right', va='top', fontsize=9)

plt.title('微博“AI看病 / AI问诊”净情感指数随时间演变')
plt.xlabel('周')
plt.ylabel('净情感指数  (-1 ~ 1)')
plt.legend()
plt.tight_layout()
plt.savefig('sentiment_index_timeline.png', dpi=300)
print('📈  图已保存为 sentiment_index_timeline.png')
