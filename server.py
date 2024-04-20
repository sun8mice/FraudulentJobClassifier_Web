from flask import Flask, render_template, request, jsonify
import os
import sys
from modeltraining import datapreprocessing
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.json['text'] # 接受前端传入的输入文本
    
    #去HTML符号
    text = datapreprocessing.html_process(text)
    print(text)
    #中文预处理
    text = datapreprocessing.chinese_text_preprocess(text)
    # 翻译
    text = datapreprocessing.translate_CH2EN(text)
    print(text)
    # 调用datapreprocess.py中的函数处理文本
    processed_text = datapreprocessing.preprocess_text(text)
    print(text)
    # 在这里调用模型进行预测
    prediction_result = datapreprocessing.predict_text(processed_text)

    return jsonify({'result': prediction_result})

if __name__ == '__main__':
    app.run(debug=True)