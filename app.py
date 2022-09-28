from flask import Flask, jsonify, request, render_template, flash, redirect, url_for
from flask_cors import CORS
import werkzeug.utils
import os
import cv2 as cv
from configs import *
from tools import *
from model_env.model import *
from model_env.logo_model import *

app = Flask(__name__)
CORS(app)

app.config['SECRET_KEY'] = SECRET_KEY
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/', methods=['GET', 'POST'])
def index():
    files = os.listdir(os.path.join('static', 'uploads'))
    print(files)
    return render_template('Home.html', files=files)


@app.route('/test', methods=['GET', 'POST'])
def test():
    return jsonify({"message": 2})


@app.route('/upload', methods=['GET', 'POST'])
def uploadfile():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect('/upload')
        file = request.files['file']
        if file.filename == '':
            flash('No image selected for uploading')
            return redirect('/upload')
        if file and allowed_file(file.filename):
            filename = file.filename.replace(' ', '_')
            print(filename)
            filename = werkzeug.utils.secure_filename(filename)
            if not os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], filename.split('.')[0])):
                os.mkdir(os.path.join(app.config['UPLOAD_FOLDER'], filename.split('.')[0]))
                os.mkdir(os.path.join(app.config['UPLOAD_FOLDER'], filename.split('.')[0], 'brand'))

            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename.split('.')[0], 'original.jpg'))

            img = cv.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename.split('.')[0], 'original.jpg'))

            xy, _ = usemodel(img.copy(), os.path.join(app.config['UPLOAD_FOLDER'], filename.split('.')[0], 'brand'))
            visualize_model(img, filename, model_finetune)


            return render_template('upload.html', filename=filename,logo = xy['name'])
        else:
            flash('Allowed image types are - png, jpg, jpeg, gif')
            return redirect('/')
    return render_template('upload.html')


@app.route('/display/<filename>/<file>')
def display_image(filename, file):
    path = os.path.join('uploads', filename.split('.')[0], file)
    return redirect(url_for('static', filename=path), code=301)


@app.route('/<filename>', methods=['GET', 'POST'])
def car(filename):
    try:
        data = {
            'logo':glob.glob(os.path.join('static', 'uploads', filename,'brand', '*.jpg'))[0]
        }
    except: data=dict()
    files = glob.glob(os.path.join('static', 'uploads', filename, '*.png'))
    files = list(map(lambda x: x.replace('static/', ''), files))
    print(files)
    return render_template('Car.html', filename=filename, files=files,data=data)


@app.route('/clear')
def clear():
    return jsonify({'status': 'remove all file success.', 'filenames': clear_file()}), 201


if __name__ == '__main__':
    app.run(debug=True, port=PORT)
