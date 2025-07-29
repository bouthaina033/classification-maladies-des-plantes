from flask import Flask, render_template, request
import os
from datetime import datetime
import image_processor as ip


from predict_rf import predict_processed_image_rf

app = Flask(__name__)
app.config.update({
    'UPLOAD_FOLDER': 'static/uploads',
    'PROCESSED_FOLDER': 'static/processed',
    'MODEL_PATH': 'svm_model.joblib'
})

@app.route('/', methods=['GET', 'POST'])
def index():
    result = {
        'original': None,
        'processed': None,
        'prediction_svm': None,
        'confidence_svm': None,
        'prediction_rf': None,
        'confidence_rf': None,
        'plante_rf': None,
        'maladie_rf': None
    }

    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', **result)

        file = request.files['file']
        if file and allowed_file(file.filename):
            # Sauvegarder l'image originale
            filename = f"orig_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            result['original'] = filename

            # Traitement de l'image
            processed_filename = ip.traiter_image_grayscale(filepath, app.config['PROCESSED_FOLDER'])
            result['processed'] = processed_filename

            # Chemin complet de l’image traitée
            processed_image_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)

            """from PIL import Image
            gray_path = os.path.join(app.config['UPLOAD_FOLDER'], f"gray_{filename}")
            try:
                img = Image.open(filepath).convert('L')  # Convertir en niveaux de gris
                img.save(gray_path)
            except Exception as e:
                print(f"Erreur conversion en niveaux de gris : {e}")
                gray_path = filepath"""

            # 🌲 Prédiction avec Random Forest
            rf_result = predict_processed_image_rf(processed_image_path)
            if rf_result:
                result['prediction_rf'] = rf_result['classe']
                result['confidence_rf'] = rf_result['confiance']
                result['plante_rf'] = rf_result['plante']
                result['maladie_rf'] = rf_result['maladie']

    return render_template('index.html', **result)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
    app.run(debug=True)
