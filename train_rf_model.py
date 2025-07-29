import os
import cv2
import numpy as np
import time
import joblib
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

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

# 🔁 Paramètres
dataset_path = r"C:/Users/aya/Downloads/archive/plantvillage dataset/grayscale"
features_file = "features_multiclass_dataset.npz"
model_file = "rf_model.joblib"
max_images_per_class = 100

start_time = time.time()

# 📥 Chargement des features
if os.path.exists(features_file):
    data = np.load(features_file)
    X = data["X"]
    y = data["y"]
else:
    print("🔄 Extraction des caractéristiques...")
    X, y = [], []
    for class_folder in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_folder)
        if not os.path.isdir(class_path):
            continue

        label = class_folder  # Nom complet du dossier = classe
        img_files = os.listdir(class_path)[:max_images_per_class]

        for img_file in img_files:
            img_path = os.path.join(class_path, img_file)
            try:
                features = extract_combined_features(img_path)
                X.append(features)
                y.append(label)
            except Exception as e:
                print(f"❌ Erreur avec {img_path} : {e}")

    X = np.array(X)
    y = np.array(y)
    np.savez_compressed(features_file, X=X, y=y)

# 🧪 Séparation train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 🌲 Entraînement Random Forest
print("🌲 Entraînement du modèle Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 💾 Sauvegarde du modèle
joblib.dump(rf, model_file)
print(f"✅ Modèle Random Forest sauvegardé sous : {model_file}")

