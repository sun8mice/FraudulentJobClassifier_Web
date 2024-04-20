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
# 逐行处理HTML文本
from bs4 import BeautifulSoup


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

#html文本预处理
def html_process(text) :
    # 使用BeautifulSoup解析HTML文本
    soup = BeautifulSoup(text, "html.parser")
    # 获取纯文本内容
    plain_text = soup.get_text(strip=True)
    return plain_text

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

#数据集预处理
def datasetprocess(data) :
    data['telecommuting']=data['telecommuting'].map({'f':0,'t':1})
    data['has_company_logo']=data['has_company_logo'].map({'f':0,'t':1})
    data['has_questions']=data['has_questions'].map({'f':0,'t':1})
    data['fraudulent']=data['fraudulent'].map({'f':0,'t':1})

    #删除一些特征由于后续用于预测中文文本，而数据集地区都是美国，故删除地区特征
    columns=['in_balanced_dataset', 'telecommuting', 'has_company_logo', 'has_questions', 'salary_range', 'employment_type','location']
    for col in columns:
        del data[col]

    #空格填充空值
    data.fillna(' ', inplace=True)
    # 去重
    data.drop_duplicates(inplace=True)
    data.reset_index(drop=True, inplace=True)

    #合并文本数据类型特征，开始数据清洗
    data['text']=(data['title']+' '+data['department']
                +' '+data['company_profile']+' '+data['description']+' '+data['requirements']
                +' '+data['benefits']+' '+data['required_experience']+' '+data['required_education']
                +' '+data['industry']+' '+data['function'])
    del data['title']
    del data['department']
    del data['company_profile']
    del data['description']
    del data['requirements']
    del data['benefits']
    del data['required_experience']
    del data['required_education']
    del data['industry']
    del data['function']

    data['text'] =   data['text'].apply(html_process)
    data['text'] = data['text'].apply(preprocess_text)


    return data

# text = "Here is an example sentence, with some punctuation! It also contains numbers like 123."
# processed_text = preprocess_text(text)
# print(processed_text)