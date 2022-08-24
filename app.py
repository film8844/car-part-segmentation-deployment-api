from flask import Flask, jsonify, request, render_template, flash, redirect, url_for
from flask_cors import CORS
import werkzeug.utils
import os
import cv2 as cv

from configs import *
from tools import *
from model_env.model import *

app = Flask(__name__)
CORS(app)

app.config['SECRET_KEY'] = SECRET_KEY
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def uploadfile():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect('/upload')
        file = request.files['file']
        if file.filename == '':
            flash('No image selected for uploading')
            return redirect('/')
        if file and allowed_file(file.filename):
            filename = file.filename.replace(' ', '_')
            print(filename)
            filename = werkzeug.utils.secure_filename(filename)
            if not os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], filename.split('.')[0])):os.mkdir(os.path.join(app.config['UPLOAD_FOLDER'], filename.split('.')[0]))

            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename.split('.')[0],'original.'+filename.split('.')[-1]))

            img = cv.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename.split('.')[0],'original.'+filename.split('.')[-1]))

            visualize_model(img,filename,model_finetune)

            flash('Image successfully uploaded and displayed below')
            return render_template('upload.html', filename=filename)
        else:
            flash('Allowed image types are - png, jpg, jpeg, gif')
            return redirect('/')
    return render_template('upload.html')


@app.route('/display/<filename>/<file>')
def display_image(filename,file):
    return redirect(url_for('static', filename='uploads/' + filename.split('.')[0]+'/'+file), code=301)


@app.route('/clear')
def clear():
    return jsonify({'status': 'remove all file success.', 'filenames': clear_file()}), 201




if __name__ == '__main__':
    app.run(debug=True, port=PORT)
