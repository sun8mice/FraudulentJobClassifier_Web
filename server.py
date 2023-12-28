from flask import Flask, render_template, request, jsonify
import os
import sys
import datapreprocess
import model_get

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
    text_vec = datapreprocess.vectorizer(processed_text)
    
    # 在这里调用模型进行预测
    model = model_get.get_pymodel()
    prediction_result = model.predict(text_vec) # 这里需要替换为实际的模型预测代码

    return jsonify({'result': prediction_result})

if __name__ == '__main__':
    app.run(debug=True)