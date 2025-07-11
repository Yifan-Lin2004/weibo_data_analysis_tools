# 微博数据分析工具包

## 项目简介

本项目围绕微博数据的采集、情感分析、主题建模和可视化展开。通过爬取微博数据，结合情感分析与LDA主题建模等技术，对不同城市、用户类型、使用设备、时间段的微博内容进行深入分析，并生成多种可视化结果。

> **说明：本项目中的微博爬虫部分基于 [dataabc/weibo-search](https://github.com/dataabc/weibo-search) 项目进行，特此致谢。**

## 环境依赖/安装方法

建议使用 Python 3.7 及以上版本。依赖的主要第三方库包括（请根据实际 requirements.txt 补充）：

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jieba wordcloud scrapy
```

## 主要功能说明（每个文件用途）

- **sumup.py**：用于将两个数据按时间顺序进行合并并进行去重处理。
- **annotate_sentiment.py**：对微博文本进行情感标注。
- **city_sentiment.py**：分析不同城市的微博情感分布。
- **lda_bert_time.py**：结合LDA和BERT进行主题建模与时间分析。
- **LDA_time.py**：基于LDA的主题建模与时间分布分析。
- **likes-comments-retweets_sentiment_topic.py**：分析点赞、评论、转发与情感、主题的关系。
- **phone_region_sentiment_analysis.py**：品牌 / 地域 / 会员类型情感交叉分析。
- **phone_sentiment.py**：手机型号的情感分析脚本。
- **sentiment_map.py**：生成情感分布地图。
- **sentiment_timeline.py**：生成情感随时间变化的可视化。
- **social_web.py**：社交网络分析相关脚本。
- **stopword.txt**：中文停用词表。
- **user_type_sentiment.py**：不同用户类型的情感分析。
- **weibo_volume_timeline.py**：微博发文量随时间变化分析。

## 使用方法（示例命令）

1. 运行微博爬虫（需借助weibo-search-master）：

   ```bash
   cd weibo-search-master
   scrapy crawl search
   ```

2. 进行情感分析：

   ```bash
   python annotate_sentiment.py
   ```

3. 生成情感地图：

   ```bash
   python sentiment_map.py
   ```

4. 主题建模与时间分析：

   ```bash
   python LDA_time.py
   ```

5. 其他脚本用法类似，具体可通过 `python 文件名.py` 运行。

