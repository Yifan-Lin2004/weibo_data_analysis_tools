import os, sys, re, io, json, requests, pandas as pd, geopandas as gpd
import matplotlib.pyplot as plt, seaborn as sns, contextily as ctx

# ----------- 路径 -----------
CSV      = "combined_annotated.csv"           # 你的微博数据
OUT_DIR  = "outputs"; os.makedirs(OUT_DIR, exist_ok=True)
MAP_PNG  = os.path.join(OUT_DIR, "province_sentiment_map.png")
BAR_PNG  = os.path.join(OUT_DIR, "region_barplot.png")

# ----------- 1. 下载省-市映射 pc.json -----------
PC_URL = ("https://raw.githubusercontent.com/modood/"
          "Administrative-divisions-of-China/master/dist/pc.json")
try:
    pc_json = json.loads(requests.get(PC_URL, timeout=30).content.decode('utf-8'))
except Exception as e:
    sys.exit(f"[❌] 无法下载 pc.json: {e}")

# 构建 city(含市/区/县) → province 映射
city_to_prov = {}
for prov, cities in pc_json.items():
    # 某些省份键带“省/市/自治区”尾缀，用正则去掉统一
    prov_short = re.sub(r"(省|市|壮族自治区|回族自治区|维吾尔自治区|自治区|特别行政区)$", "", prov)
    city_to_prov[prov_short] = prov_short  # 省会或直辖市本身
    for city in cities:
        city_short = re.sub(r"(市|区|县|自治州|盟|地区|特别行政区)$", "", city)
        city_to_prov[city_short] = prov_short

# 省正则
prov_names = sorted({v for v in city_to_prov.values()}, key=len, reverse=True)
prov_pattern = re.compile("|".join(prov_names))

# ----------- 2. 微博数据读取 -----------
df = pd.read_csv(CSV)
need_cols = {'ip','sent_score'}
if not need_cols.issubset(df.columns):
    sys.exit(f"[❌] CSV 缺失列: {need_cols - set(df.columns)}")

def loc_to_prov(loc: str) -> str:
    """优先匹配城市，再回退直接匹配省名"""
    if pd.isna(loc) or not str(loc).strip():
        return "未知/海外"
    loc = str(loc)
    # 城市匹配
    for city in city_to_prov:
        if city in loc:
            return city_to_prov[city]
    # 省匹配
    m = prov_pattern.search(loc)
    return m.group() if m else "未知/海外"

df['省份'] = df['ip'].apply(loc_to_prov)

# ----------- 3. 省份 → 大区 -----------
region_map = {
    "北京 天津 上海 重庆":                "直辖市",
    "河北 山西 河南 陕西":                "华北",
    "辽宁 吉林 黑龙江":                   "东北",
    "江苏 浙江 安徽 福建 山东 江西":       "华东",
    "湖北 湖南 广东 广西 海南 香港 澳门 台湾": "华南",
    "四川 贵州 云南":                     "西南",
    "甘肃 青海 宁夏 新疆 西藏 内蒙古":       "西北",
}
prov2reg = {p:reg for provs,reg in region_map.items() for p in provs.split()}
df['大区'] = df['省份'].map(prov2reg).fillna("未知/海外")

# ----------- 4. 省级情感均值 ----------
prov_mean = (df.groupby('省份')['sent_score']
               .mean()
               .rename('sent_mean')
               .reset_index())

# ----------- 5. 省界 GeoJSON ----------
GEO_URL = "https://geo.datav.aliyun.com/areas_v3/bound/geojson?code=100000_full"
china = gpd.read_file(GEO_URL)[['name','geometry']].rename(columns={'name':'省份'})
china['省份'] = china['省份'].str.replace("省|市|壮族自治区|回族自治区|维吾尔自治区|自治区|特别行政区", "", regex=True)
gdf = china.merge(prov_mean, on='省份', how='left')

# ----------- 6. 热力图 ----------
fig, ax = plt.subplots(figsize=(8,6))
gdf.boundary.plot(ax=ax, color='gray', linewidth=.3)
gdf.plot(column='sent_mean', cmap='RdYlGn', vmin=-1, vmax=1,
         edgecolor='gray', linewidth=.2, legend=True, ax=ax)
try:
    ctx.add_basemap(ax, crs=gdf.crs, source=ctx.providers.Stamen.TonerLite)
except Exception:
    print("[WARN] 加载底图失败，已跳过 basemap")
ax.set_title("省级微博情感均值热力图"); ax.axis('off')
plt.tight_layout(); plt.savefig(MAP_PNG, dpi=300); plt.close()
print("✅ 省级地图已保存:", MAP_PNG)

# ----------- 7. 大区柱状图 ----------
stats = (df[df['大区']!="未知/海外"]
         .groupby('大区')['sent_score']
         .agg(['mean','std','count']))
stats['ci95'] = 1.96 * stats['std'] / stats['count']**0.5
stats = stats.sort_values('mean', ascending=False)

plt.figure(figsize=(7,4))
sns.barplot(x=stats.index, y=stats['mean'],
            yerr=stats['ci95'], capsize=.2, color='#5B8FF9')
plt.ylabel("平均情感分数"); plt.xticks(rotation=30)
plt.title("各大区情感均值 ±95%CI")
plt.tight_layout(); plt.savefig(BAR_PNG, dpi=300); plt.close()
print("✅ 大区柱状图已保存:", BAR_PNG)
