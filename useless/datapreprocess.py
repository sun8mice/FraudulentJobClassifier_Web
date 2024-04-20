import joblib
import jieba
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

import requests
import hashlib
import random

# 测试用
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

# 英文文本预处理
def preprocess_text(text):
    # 去除特殊字符、标点符号和数字
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # 转换为小写
    text = text.lower()
    # 分词
    words = word_tokenize(text)
    # 去除停用词
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    # 词干提取
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    # 拼接处理后的单词为字符串
    processed_text = ' '.join(words)
    return processed_text

def get_stacking_model():  # 获取Stacking模型
    model_path = 'static/model/stacking_model.pkl'
    model = joblib.load(model_path)
    return model

def get_vectorizer():  # 获取向量化器
    vectorizer_path = 'static/vectorizer/tfidfvectorizer'
    vectorizer = joblib.load(vectorizer_path)
    return vectorizer

def predict_text(text):
    # 加载模型和向量化器
    stacking_model = get_stacking_model()
    vectorizer = get_vectorizer()
    
    # 将文本向量化
    text_vectorized = vectorizer.transform([text])
    
    # 进行预测
    prediction = stacking_model.predict(text_vectorized)
    
    result = check_result(prediction)

    return result

def check_senti(prediction_result):
        # 将预测结果转换为列表
    prediction_result = prediction_result.tolist()
    # 判断是正面还是负面
    if prediction_result[0] == 1:
        sentiment = '正面'
    else:
        sentiment = '负面'
    return sentiment

# 调用百度api进行翻译
def translate_CH2EN(text):  
    # 定义百度翻译 API 的应用ID和密钥
    app_id = '20240419002029283'
    app_key = '1mFglrMwapSvc7KYpvQ9'

    # 源语言和目标语言，这里示例为中文到英文
    from_lang = 'zh'
    to_lang = 'en'

    # 生成随机数
    salt = random.randint(32768, 65536)

    # 计算签名
    sign = app_id + text + str(salt) + app_key
    sign = hashlib.md5(sign.encode()).hexdigest()

    # 构造请求参数
    data = {
        'q': text,
        'from': from_lang,
        'to': to_lang,
        'appid': app_id,
        'salt': salt,
        'sign': sign
    }

    try:
        # 发送请求
        response = requests.get('http://api.fanyi.baidu.com/api/trans/vip/translate', params=data)
        result = response.json()

        # 检查是否存在翻译结果
        if 'trans_result' not in result or len(result['trans_result']) == 0:
            error_code = result.get('error_code', 'Unknown')
            error_msg = result.get('error_msg', 'Unknown error')
            raise Exception(f"Baidu API Error {error_code}: {error_msg}")

        # 提取翻译结果
        translated_text = result['trans_result'][0]['dst']

        return translated_text

    except Exception as e:
        raise Exception(f"Baidu API Error: {e}")

def check_result(prediction_result):
    # 将预测结果转换为列表
    prediction_result = prediction_result.tolist()
    # 判断是正面还是负面
    if prediction_result[0] == 1:
        sentiment = '该招聘信息不值得信任'
    else:
        sentiment = '该招聘信息值得信任'
    return sentiment