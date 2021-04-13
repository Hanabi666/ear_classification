from flask import Flask,request,jsonify
from utils import *
from detection import *
import io
import uuid
import traceback
import logging


logging.basicConfig(filename='.log/log.txt', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

@app.route('/')
def hello():
    return '<h1>Flask Test</h1>'

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        """
        with open('testfile.txt', 'a') as f:
            print(request.files, file=f)
            print(request.form, file=f)
            print(request.data, file=f)
            print('*'*20, file=f)

        """
        try:
            f = request.files['imagetype']
            image = f.read()
            image = Image.open(io.BytesIO(image))
            output = predict(image)
            Json = list_to_json(output)
            uid = uuid.uuid1()
            image.save('.log/img/'+str(uid)+'.jpg')
            with open('.log/json/'+str(uid)+'.json', 'w', encoding='utf8') as f:
                json.dump(Json, f, ensure_ascii=False)
        except:
            logging.debug(traceback.format_exc())
            
        return Json
        
        #return '<h1>ok</h1>'
    
    else:
        return '<h1>The method should be POST</h1>'

if __name__ == '__main__':
    app.run()

# test command: curl -X POST -F file=@/home/zwzwzw/Flask/classification_demo/test/normal.png http://127.0.0.1:5000/upload
