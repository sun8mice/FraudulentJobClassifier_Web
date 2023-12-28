import pandas as pd
import jieba
# 导入情感分析所需的库
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import datapreprocess
import joblib

# 导入数据集
data = pd.read_csv('./temp/item_comments.csv')

# 对text列进行分词和停用词过滤
data['evaluation'] = data['evaluation'].apply(datapreprocess.preprocess)

# 划分训练集和测试集
X = data['evaluation']  # 特征
y = data['label']  # 标签
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征提取
X_train_vec = datapreprocess.vectorizer(X_train)
X_test_vec = datapreprocess.vectorizer(X_test)

# 建立模型
model = MultinomialNB()
model.fit(X_train_vec, y_train)

joblib.dump(model, './static/model/model.pkl')