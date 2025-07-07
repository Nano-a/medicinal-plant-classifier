import os
from flask import Flask, render_template
from controller.routes import main

# Configuration de Flask
app = Flask(__name__,
            template_folder='templates',
            static_folder='static')

# Configuration des chemins
app.config['UPLOAD_FOLDER'] = os.path.abspath('data')
app.config['STATIC_FOLDER'] = os.path.abspath('static')
app.config['VISUALIZATIONS_FOLDER'] = os.path.abspath('visualizations')

# Enregistrer le blueprint
app.register_blueprint(main)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    # Créer les dossiers nécessaires s'ils n'existent pas
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)
    os.makedirs(app.config['VISUALIZATIONS_FOLDER'], exist_ok=True)
    
    app.run(debug=True, port=5001)
