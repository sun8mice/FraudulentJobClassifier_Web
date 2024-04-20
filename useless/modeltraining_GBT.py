import pandas as pd

# 数据导入和token映射
file_path = './src/dataset/DataSet.csv' 
data = pd.read_csv(file_path)
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

# 逐行处理HTML文本
from bs4 import BeautifulSoup
for i in range(len(data)):
    html_text = data.loc[i, 'text']
    # 使用BeautifulSoup解析HTML文本
    soup = BeautifulSoup(html_text, "html.parser")
    # 获取纯文本内容
    plain_text = soup.get_text(strip=True)
    # 将处理后的文本保存回数据集
    data.loc[i, 'text'] = plain_text

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
nltk.download('punkt')

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

# 预处理文本特征列
data['clean_text'] = data['text'].apply(preprocess_text)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data['clean_text'], data['fraudulent'], test_size=0.2, random_state=42)

# TF-IDF向量化
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.metrics import classification_report, accuracy_score

# 创建梯度提升树分类器
gb_classifier = GradientBoostingClassifier()

param_dist = {
    'n_estimators': 162,        # 决策树数量
    'max_depth': 6,             # 树的最大深度
    'min_samples_split': 6,     # 内部节点再划分所需最小样本数
    'min_samples_leaf': 7,      # 叶子节点最少样本数
    'learning_rate': 0.2,       # 学习率
    'subsample': 1.0,           # 使用所有样本进行训练
    'loss': 'log_loss',         # 使用对数损失函数
    'min_impurity_decrease': 0.1,  # 控制正则化
}

# 创建梯度提升树分类器，并设置最佳参数
gb_classifier = GradientBoostingClassifier(**param_dist)

# 在训练数据上拟合模型
gb_classifier.fit(X_train_tfidf, y_train)
y_pred = gb_classifier.predict(X_test_tfidf)

print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# 导出训练好的模型
import joblib
joblib.dump(gb_classifier, 'static/model/gbt_model.pkl')
# 导出TF-IDF向量化器
joblib.dump(tfidf_vectorizer, 'static/vectorizer/tfidf_vectorizer.pkl')