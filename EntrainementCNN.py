import logging
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import layers, models
import numpy as np
import os
from tensorflow.keras.utils import img_to_array, load_img
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# Chemins des dossiers
train_dir = r"C:\Users\coren\MonDrive\Projet\Entrainement"

# Taille des images et paramètres
img_size = (224, 224)
batch_size = 32
epochs = 10

# Charger les données d'entraînement + validation (80/20 split)
validation_split = 0.2
seed = 123
train_dataset = image_dataset_from_directory(
    directory=train_dir,
    labels="inferred",
    label_mode="binary",
    batch_size=batch_size,
    image_size=img_size,
    shuffle=True,
    validation_split=validation_split,
    subset="training",
    seed=seed
)

val_dataset = image_dataset_from_directory(
    directory=train_dir,
    labels="inferred",
    label_mode="binary",
    batch_size=batch_size,
    image_size=img_size,
    shuffle=False,
    validation_split=validation_split,
    subset="validation",
    seed=seed
)

# Normalisation des images
normalization_layer = layers.Rescaling(1./255)
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y)).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y)).prefetch(tf.data.AUTOTUNE)

# Construire le modèle CNN
inputs = tf.keras.Input(shape=(224, 224, 3))

x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
x = layers.MaxPooling2D((2, 2))(x)

x = layers.Conv2D(64, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)

# On nomme la dernière couche conv (important pour GradCAM)
x = layers.Conv2D(128, (3, 3), activation='relu', name="last_conv")(x)
x = layers.MaxPooling2D((2, 2))(x)

# ⚠️ ON REMPLACE FLATTEN PAR GLOBAL AVERAGE POOLING
x = layers.GlobalAveragePooling2D()(x)

x = layers.Dense(128, activation='relu')(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model_path = os.path.join(os.getcwd(), 'reseauncnn_best.keras')

# Callbacks: checkpoint, csv logger, early stopping
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss'),
    tf.keras.callbacks.CSVLogger('reseauncnn_training_log.csv'),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
]

logging.info("Entraînement du modèle...")
history = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs, callbacks=callbacks, verbose=1)

# Sauvegarde finale (optionnelle) — déjà sauvegardé par ModelCheckpoint
final_model_path = os.path.join(os.getcwd(), 'reseauncnn_final.keras')
try:
    model.save(final_model_path)
    logging.info(f"Model saved to {final_model_path}")
except Exception:
    logging.warning("Could not save final model:\n" + __import__('traceback').format_exc())

# Plot training history (accuracy & loss)
try:
    acc = history.history.get('accuracy') or history.history.get('acc')
    val_acc = history.history.get('val_accuracy') or history.history.get('val_acc')
    loss = history.history.get('loss')
    val_loss = history.history.get('val_loss')

    if val_acc:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(acc, label='train_accuracy')
        plt.plot(val_acc, label='val_accuracy')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        if loss and val_loss:
            plt.plot(loss, label='train_loss')
            plt.plot(val_loss, label='val_loss')
            plt.title('Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)

        plt.tight_layout()
        plt.show()

        best_val = max(val_acc)
        best_epoch = val_acc.index(best_val) + 1
        print(f"Best val_accuracy: {best_val:.4f} at epoch {best_epoch}")
    else:
        logging.info("No validation accuracy available in history.")
except Exception as e:
    logging.warning("Could not plot training history: " + str(e))

# Test du modèle
def predict_article(article_path):
    """
    Prédire si un article est Legit ou Fake à partir de ses images,
    et donner la confiance en pourcentage.
    """
    images = []
    probabilities = []
    
    for img_file in os.listdir(article_path):
        img_path = os.path.join(article_path, img_file)
        if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
            img = load_img(img_path, target_size=img_size)
            img_array = img_to_array(img) / 255.0
            images.append(img_array)

    if not images:
        print(f"Aucune image trouvée dans {article_path}")
        return None

    images = np.array(images)
    predictions = model.predict(images)
    
    # Calcul des probabilités et confiance
    for i, pred in enumerate(predictions):
        prob = pred[0]  # La probabilité pour cette image d'être Legit
        probabilities.append(prob)
    
    # Compter les occurrences de Legit et Fake en fonction des probabilités
    fake_count = np.sum(np.array(probabilities) < 0.5)
    legit_count = np.sum(np.array(probabilities) >= 0.5)
    
    # Calculer la probabilité moyenne
    avg_confidence = np.mean(probabilities) * 100  # En pourcentage
    result = "Legit" if legit_count > fake_count else "Fake"
    
    print(f"Confiance moyenne: {avg_confidence:.2f}%")
    return result, fake_count, legit_count, avg_confidence


