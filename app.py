from flask import Flask,request,jsonify
from utils import *
from detection import *
import io


app = Flask(__name__)

@app.route('/hello')
def hello():
    return '<h1>Hello World</h1>'

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        image = f.read()
        image = Image.open(io.BytesIO(image))
        output = predict(image)
        Json = list_to_json(output)
        return Json
        #return jsonify({'class_id': '0', 'class_name': '1'})

if __name__ == '__main__':
    app.run()

# test command: curl -X POST -F file=@/home/zwzwzw/Flask/classification_demo/test/normal.png http://127.0.0.1:5000/upload