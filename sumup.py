#用于将两个数据按时间顺序进行合并并进行去重处理
import pandas as pd

# 1. 读入两个表格
df_kanbing = pd.read_csv('weibo-search-master/结果文件/AI看病/AI看病.csv')
df_wenzhen = pd.read_csv('weibo-search-master/结果文件/AI问诊/AI问诊.csv')
combined = pd.concat([df_kanbing, df_wenzhen], ignore_index=True)
combined['发布时间'] = pd.to_datetime(combined['发布时间'], errors='coerce')
combined = combined.drop_duplicates()
combined = combined.sort_values('发布时间', ascending=True)

combined.to_csv('combined.csv', index=False, encoding='utf-8-sig')

print('合并、去重、排序完成，已保存为 combined.csv')

