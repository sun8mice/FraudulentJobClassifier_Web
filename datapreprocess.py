import jieba
import pandas as pd

def preprocess(text):
    # 加载停用词表
    stopwords = pd.read_csv('./stopWords/stop_words.txt', index_col=False, quoting=3, sep="\t", names=['stopword'], encoding='utf-8')
    # 转换为列表
    stopwords_list = stopwords['stopword'].tolist()  
    # 分词
    seg_list = jieba.cut(text)
    # 去除停用词
    filtered_words = [word for word in seg_list if word not in stopwords_list]
    # 返回处理后的分词结果
    return ' '.join(filtered_words) 

