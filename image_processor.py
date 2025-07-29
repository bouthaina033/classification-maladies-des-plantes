from PIL import Image, ImageFilter, ImageEnhance, ImageOps
from datetime import datetime
import os
import numpy as np
import cv2


def unsharp_mask(image, sigma=1.0, strength=1.5, threshold=0):

    # Créer une version floue de l'image originale
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)

    # Calculer le masque (différence entre original et flou)
    mask = cv2.subtract(image, blurred)

    # Appliquer le seuil si spécifié
    if threshold > 0:
        mask = np.where(np.abs(mask) < threshold, 0, mask)

    # Ajouter le masque amplifié à l'image originale
    sharpened = cv2.addWeighted(image, 1.0, mask, strength, 0)

    # S'assurer que les valeurs restent dans la plage [0, 255]
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    return sharpened


def traiter_image_grayscale_avec_unsharp(image_path, processed_folder, width=256, height=256, use_unsharp=True):

    try:
        # Charger l'image couleur avec OpenCV
        image = cv2.imread(image_path)
        if image is None:
            raise Exception("Impossible de lire l'image")

        # Convertir en niveaux de gris
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if use_unsharp:
            # NOUVEAU: Traitement avec Unsharp Mask

            # Débruitage léger préalable
            filtered = cv2.medianBlur(gray, 3)

            # Appliquer l'unsharp mask avec paramètres optimisés pour les feuilles
            enhanced = unsharp_mask(
                filtered,
                sigma=2.0,  # Flou gaussien modéré
                strength=1.2,  # Amélioration modérée
                threshold=5  # Seuil pour réduire le bruit
            )

            # Ajustement optionnel du contraste
            img_array = np.array(enhanced, dtype=np.float64)
            img_array = np.clip((img_array - 128) * 1.1 + 128, 0, 255)

        else:
            # ANCIEN: Traitement original

            # Appliquer un filtre médian pour supprimer le bruit
            filtered = cv2.medianBlur(gray, 5)

            # Amélioration des bords avec un filtre Laplacien
            laplacian = cv2.Laplacian(filtered, cv2.CV_64F)
            laplacian = cv2.convertScaleAbs(laplacian)

            # Fusionner l'image filtrée et ses détails (renforcement)
            enhanced = cv2.addWeighted(filtered, 0.8, laplacian, 0.2, 0)

            # Convertir en array pour manipulations supplémentaires
            img_array = np.array(enhanced, dtype=np.float64)

            # Ajustement gamma pour assombrir légèrement
            gamma = 0.9
            img_array = 255 * (img_array / 255) ** gamma

            # Augmenter le contraste
            img_array = np.clip((img_array - 128) * 1.2 + 128, 0, 255)

        # Convertir en image PIL pour les étapes suivantes
        img_final = Image.fromarray(img_array.astype(np.uint8))

        # Redimensionner l'image si nécessaire
        image_processed = img_final.resize((width, height), Image.Resampling.LANCZOS)

        # Génération du nom de fichier avec indication du traitement
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        method_suffix = "unsharp" if use_unsharp else "laplacian"
        processed_filename = f"processed_{method_suffix}_{timestamp}.png"
        save_path = os.path.join(processed_folder, processed_filename)

        # Sauvegarde
        image_processed.save(save_path)
        method_name = "Unsharp Mask" if use_unsharp else "Laplacien"
        print(f"Image traitée ({method_name}) sauvegardée dans : {save_path}")

        return processed_filename

    except Exception as e:
        print(f"Erreur lors du traitement de l'image : {e}")
        return None


def traiter_image_grayscale(image_path, processed_folder, width=256, height=256):

    return traiter_image_grayscale_avec_unsharp(
        image_path, processed_folder, width, height, use_unsharp=True
    )







