import cv2
import numpy as np
import joblib
from skimage.feature import hog


def extract_combined_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Image introuvable : {image_path}")

    img_resized = cv2.resize(img, (64, 64))

    hist = cv2.calcHist([img_resized], [0], None, [32], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    hog_features = hog(img_resized, orientations=6, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)

    mean_val = np.mean(img_resized)
    std_val = np.std(img_resized)

    return np.hstack([hist, hog_features, mean_val, std_val])


def predict_processed_image_rf(processed_image_path):
    try:
        model = joblib.load("rf_model.joblib")
        features = extract_combined_features(processed_image_path).reshape(1, -1)

        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        confidence = round(np.max(probabilities) * 100, 2)

        # Séparation de la classe prédite
        if "___" in prediction:
            plant_type, disease = prediction.split("___")
        else:
            plant_type = prediction
            disease = "unknown"

        return {
            'classe': prediction,
            'plante': plant_type,
            'maladie': disease,
            'confiance': confidence
        }

    except Exception as e:
        print(f"Erreur de prédiction RF : {str(e)}")
        return None
