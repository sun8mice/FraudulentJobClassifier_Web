import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
import joblib
# import datapreprocess
import jieba

#读取数据，映射标签
df = pd.read_csv("temp/item_comments.csv")
df["label"] = df["label"].map({"正面": 1, "负面": 0})

#分词，剔除停用词
words= []
for i,row in  df.iterrows():
    word = jieba.cut(row['evaluation'])
    result = '  '.join(word)
    words.append(result)

#向量化
vect = CountVectorizer() #将文本转换为数值，构成特征向量
X = vect.fit_transform(words)
X = X.toarray()
y = df['label']

#划分训练集测试集
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.1,random_state = 1)

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier()
mlp.fit(X_train,y_train)

# y_pred = mlp.predict(X_test)
# from sklearn.metrics import accuracy_score
# score = accuracy_score(y_pred,y_test)
# print(score)

# comment = input("请输入你对商品的评价：")
# comment = datapreprocess.preprocess(comment)
# print(comment)
# X_try = vect.transform(comment)
# y_pred = mlp.predict(X_try.toarray())
# print(y_pred)

# 导出模型和向量化器
joblib.dump(mlp, 'static/model/mlp_model.pkl')
joblib.dump(vect, 'static/vectorizer/count_vect.pkl')