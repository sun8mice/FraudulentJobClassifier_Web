import joblib
import jieba
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

def preprocess(text):
    # 加载停用词表
    stopwords = pd.read_csv('stopWords/stop_words.txt', index_col=False, quoting=3, sep="\t", names=['stopword'], encoding='utf-8')
    # 转换为列表
    stopwords_list = stopwords['stopword'].tolist()
    #分词并剔除停用词
    # 分词
    words = jieba.lcut(text)
    # 剔除停用词
    words = [word for word in words if word not in stopwords_list]
    # 以空格连接词语，得到一个字符串
    result = [' '.join(words)]
    # 返回处理后的分词结果
    return result

def get_pymodel():  #获取模型
    model_path = 'static/model/mlp_model.pkl'
    model = joblib.load(model_path)
    return model

def get_vectorizer():  #获取向量化器
    model_path = 'static/vectorizer/count_vect.pkl'
    model = joblib.load(model_path)
    return model

def check_senti(prediction_result):
        # 将预测结果转换为列表
    prediction_result = prediction_result.tolist()
    # 判断是正面还是负面
    if prediction_result[0] == 1:
        sentiment = '正面'
    else:
        sentiment = '负面'
    return sentiment