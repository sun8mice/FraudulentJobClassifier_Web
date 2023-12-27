from flask import Flask, render_template, request, jsonify
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import datapreprocess

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.json['text']
    processed_text = datapreprocess.preprocess(text) # 调用datapreprocess.py中的函数处理文本
    
    # 在这里调用模型进行预测
    # prediction_result = model.predict(processed_text)  # 这里需要替换为实际的模型预测代码
    prediction_result = "前方应出现文本预处理的结果" #前期测试用

    return jsonify({'processed_text': processed_text, 'result': prediction_result})

if __name__ == '__main__':
    app.run(debug=True)