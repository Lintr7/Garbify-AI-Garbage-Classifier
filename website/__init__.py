from flask import Flask
import os

def create_app():
    app = Flask(__name__)
    
    

    app.config['SECRET_KEY'] = 'DSDSsdsd sdsd'
    app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'uploads')

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    from .views import views
    app.register_blueprint(views, url_prefix = '/')

    return app