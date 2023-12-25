from flask import Flask

#用当前脚本名称实例化Flask对象，方便flask从该脚本文件中获取需要的内容
app = Flask(__name__)

#程序实例需要知道每个url请求所对应的运行代码是谁。
#所以程序中必须要创建一个url请求地址到python运行函数的一个映射。
#处理url和视图函数之间的关系的程序就是"路由"，在Flask中，路由是通过@app.route装饰器(以@开头)来表示的
#methods参数用于指定允许的请求格式
#常规输入url的访问就是get方法
@app.route('/')
def hello_world():
    return 'Hello, World!'

app.run()
