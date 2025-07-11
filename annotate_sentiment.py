# -*- coding: utf-8 -*-
import pandas as pd, torch, os
from transformers import pipeline
from tqdm import tqdm

RAW_CSV = 'combined_sorted.csv'
OUT_CSV = 'combined_annotated.csv'

if os.path.exists(OUT_CSV):
    print(f"🚫  {OUT_CSV} 已存在，跳过推理。若想重跑请先删除该文件。")
    exit()

# ---------- 1. 读数据 ----------
df = pd.read_csv(RAW_CSV)
df['发布时间'] = pd.to_datetime(df['发布时间'], errors='coerce')

# ---------- 2. 初始化模型 ----------
model_name = 'IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment'
device_id  = 0 if torch.cuda.is_available() else -1
print(f"🚀  Using CUDA: {torch.cuda.is_available()}  (device={device_id})")

clf = pipeline(
    task='sentiment-analysis',
    model=model_name,
    tokenizer=model_name,
    device=device_id,
    batch_size=32 if device_id != -1 else 8,
    padding=True, truncation=True, max_length=512,
    return_all_scores=True         # 返回两个标签的概率
)

# ---------- 3. 批量推理 ----------
scores = []
texts = df['微博正文'].fillna('').tolist()
BATCH = 32 if device_id != -1 else 8

for i in tqdm(range(0, len(texts), BATCH), desc="🔍  calculating score"):
    batch_out = clf(texts[i:i+BATCH])       # list[list[dict]]

    for res in batch_out:
        # 转成 {label:prob}
        score_dict = {d['label'].lower(): d['score'] for d in res}
        pos = score_dict.get('positive', score_dict.get('label_1', 0.0))
        neg = score_dict.get('negative', score_dict.get('label_0', 0.0))
        scores.append(pos - neg)            # 连续得分 [-1,1]

df['sent_score'] = scores
df.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
print(f"✅  连续情感分数已写入 {OUT_CSV}")
