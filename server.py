from flask import Flask, render_template, request, jsonify
import os
import sys
import datapreprocess
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.json['text'] # 接受前端传入的输入文本
    
    # 调用datapreprocess.py中的函数处理文本
    processed_text = datapreprocess.preprocess(text)

    # 在这里调用模型进行预测
    model = datapreprocess.get_pymodel()
    vect = datapreprocess.get_vectorizer()

    X_test_web = vect.transform(processed_text)
    prediction_result = model.predict(X_test_web.toarray()) # 这里需要替换为实际的模型预测代码
    
    sentiment = datapreprocess.check_senti(prediction_result)

    return jsonify({'result': sentiment})

if __name__ == '__main__':
    app.run(debug=True)