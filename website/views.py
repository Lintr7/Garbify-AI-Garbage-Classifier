from flask import Blueprint, render_template, request, flash, redirect, current_app
from werkzeug.utils import secure_filename
from .predictImage import predict_image
import requests
import os

views = Blueprint('views', __name__)

UPLOAD_FOLDER = 'website/Static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@views.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        
        if 'image' not in request.files:
            flash('No file part in the request.')
            return redirect(request.url)
        file = request.files['image']
        
        if file.filename == '':
            flash('No file selected.')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            
            filename = secure_filename(file.filename)
            
            filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            result = predict_image(filepath)
            os.remove(filepath)
            return render_template('result.html', result=result)
            
        else:
            return render_template('invalidFile.html')
    
    return render_template('baseupload.html')

@views.route('/uploads/<filename>')
def send_uploaded_file(filename=''):
    from  flask import send_from_directory
    return send_from_directory(current_app.config['UPLOAD_FOLDER'], filename)

def refile():
    if (allowed_file):
        return render_template('invalidFile.html')
