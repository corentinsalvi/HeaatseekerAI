import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from GradCam import process_image, get_conv_layer_names  # Import des fonctions Grad-CAM

# Chemins
model_path = "reseauncnn_best.keras"
test_dir = "images/"
gradcam_out_dir = "gradcam_outputs"  # dossier pour stocker les heatmaps

# Charger le modèle
model = load_model(model_path)
IMG_HEIGHT, IMG_WIDTH = 224, 224  # à adapter à ton modèle

# Identifier les couches convolutionnelles candidates
conv_candidate_names = get_conv_layer_names(model)
print("Conv2D candidates for Grad-CAM:", conv_candidate_names)

def predict_article(article_path):
    """
    Prédit toutes les images dans un dossier d'article et génère Grad-CAM
    """
    fake_count = 0
    legit_count = 0
    confidences = []

    # Créer un dossier spécifique pour les heatmaps de cet article
    article_name = os.path.basename(article_path.rstrip(os.sep))
    article_gradcam_dir = os.path.join(gradcam_out_dir, article_name)
    os.makedirs(article_gradcam_dir, exist_ok=True)

    for img_file in sorted(os.listdir(article_path)):
        img_path = os.path.join(article_path, img_file)
        if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        # Chargement image
        img = load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
        img_array = img_to_array(img) / 255.0
        img_tensor = np.expand_dims(img_array, axis=0)

        # Prédiction
        pred = model.predict(img_tensor, verbose=0)[0][0]
        confidences.append(pred * 100)
        if pred >= 0.5:
            fake_count += 1
        else:
            legit_count += 1

        # Grad-CAM
        try:
            heatmap, out_file = process_image(
                img_path,
                model,
                target_size=(IMG_HEIGHT, IMG_WIDTH),
                conv_candidate_names=conv_candidate_names,
                out_dir=article_gradcam_dir,
                alpha=0.4
            )
        except Exception as e:
            print(f"Could not generate Grad-CAM for {img_file}: {e}")
            continue

    avg_confidence = np.mean(confidences) if confidences else 0
    result = "FAKE" if fake_count > legit_count else "LEGIT"

    return result, fake_count, legit_count, avg_confidence

# Parcourir tous les articles
os.makedirs(gradcam_out_dir, exist_ok=True)
for article_folder in sorted(os.listdir(test_dir)):
    article_path = os.path.join(test_dir, article_folder)
    if os.path.isdir(article_path):
        result, fake_count, legit_count, avg_confidence = predict_article(article_path)
        if avg_confidence > 50:
            print(f"Article {article_folder}: {result} (Legit: {legit_count}, Fake: {fake_count}, Confiance moyenne: {avg_confidence:.2f}%)")
        else: 
            print(f"L'authentification de l'article {article_folder} a une confiance moyenne de {avg_confidence:.2f}%.\nL'article semble {result} mais la confiance est faible.\nVeuillez envoyer plus d'images pour une meilleure évaluation.")