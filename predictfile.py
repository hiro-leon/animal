import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename

from keras.models import Sequential, load_model
import keras, sys
import numpy as np
from PIL import Image

classes = ['cat', 'penguin', 'zebra']
num_classes = len(classes)
image_size = 50

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'gif']) #アップロードできる拡張子を制限する

# appという名前でインスタンス化(newする)
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#ファイルが条件を満たしているかを確認するための関数
def allowed_file(filename):
    #条件が両方OKなら１,NGなら0を返す
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

#ウェブアプリケーション用のルーティング処理
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('ファイルがありません')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('ファイルがありません')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename) #危険な文字を処理
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            model =load_model('./animal_cnn_aug.h5')

            image = Image.open(filepath)
            image = image.convert('RGB')
            image = image.resize((image_size, image_size))
            data = np.asarray(image)
            X = []
            X.append(data)
            X = np.array(X)

            result = model.predict([X])[0]
            predicted = result.argmax()  # 一番推定確率が高い種類の動物を取り出す
            percentage = int(result[predicted] * 100)

            return 'ラベル: ' + classes[predicted] + ', 確率: ' + str(percentage) + '%'



            #return redirect(url_for('uploaded_file', filename = filename)) リダイレクトをしてファイルを出力する
    return '''
    <!doctype html>
    <html>
    <head>
    <meta charset="UTF-8">
    <title>ファイルをアップロードして判定しよう</title></head>
    <body>
    <h1>ファイルをアップロードして判定しよう</h1>
    <form method = post enctype = multipart/form-data>
    <p><input type=file name=file>
    <input type =submit value=Upload>
    </form>
    </body>
    </html> 
    '''

from flask import send_from_directory

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)