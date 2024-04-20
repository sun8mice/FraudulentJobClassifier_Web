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
#重设索引，避免去重后导致空索引错误
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

# 文本特征处理与模型训练

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

# 定义支持向量机模型
svm_model = SVC()

# 设置参数网格
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']}

# 使用网格搜索进行参数调优
grid_search = GridSearchCV(svm_model, param_grid, refit=True, verbose=3)
grid_search.fit(X_train_tfidf, y_train)

# 输出最佳参数
print("Best Parameters:", grid_search.best_params_)

# 在测试集上评估模型
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_tfidf)

# 输出模型评估结果
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# 导出训练好的模型
import joblib
joblib.dump(best_model, 'static/model/svm_model.pkl')
# 导出TF-IDF向量化器
joblib.dump(tfidf_vectorizer, 'static/vectorizer/tfidf_vectorizer.pkl')