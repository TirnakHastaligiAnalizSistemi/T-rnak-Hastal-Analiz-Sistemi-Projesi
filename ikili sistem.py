# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import json
import matplotlib.pyplot as plt
from tensorflow.keras.applications.densenet import preprocess_input
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score

# Modelleri yüklenmesi
binary_model = tf.keras.models.load_model("best_densenet121binary.keras")
multiclass_model = tf.keras.models.load_model("best_densenet121.keras")
BINARY_THRESHOLD = 0.4
print("Modeller yüklendi")
train_dir = r"C:\Users\metin\OneDrive - CAKABEY OKULLARI\Masaüstü\data\data\train"

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=preprocess_input
)

data_gen = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=1,
    class_mode="categorical",
    shuffle=False
)

class_names = list(data_gen.class_indices.keys())
disease_names = class_names[1:]

# İkili Sistem Modeli
y_true = []
y_pred = []

for i in range(len(data_gen)):
    img, label = data_gen[i]
    true_class = np.argmax(label)
    binary_probs = binary_model.predict(img, verbose=0)[0]
    harmful_prob = binary_probs[1]
    if harmful_prob < BINARY_THRESHOLD:
        continue

    multiclass_probs = multiclass_model.predict(img, verbose=0)[0]
    multiclass_probs[0] = 0.0
    pred_class = np.argmax(multiclass_probs)

    if true_class != 0:
        y_true.append(true_class - 1)
        y_pred.append(pred_class - 1)

print("İkili sistem tamamlandı")

# Karışıklık Matrisi
print("\nClassification Report (5 Hastalık):\n")
print(classification_report(y_true, y_pred, target_names=disease_names))
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=disease_names
)
disp.plot(xticks_rotation=45)
plt.title("İkili Sistem – 5 Sınıf Karısıklılık Matrisi")
plt.tight_layout()
plt.show()

acc = accuracy_score(y_true, y_pred)
print(f"İkili Sistem Dogrulugu: %{acc * 100:.2f}")

# Sistemi kaydetme
cascade_config = {
    "binary_model": "binary_model_final.keras",
    "multiclass_model": "multiclass_model_final.keras",
    "binary_threshold": BINARY_THRESHOLD,
    "ignored_multiclass_index": 0,
    "disease_class_count": 5,
    "description": "İkili Sistem"
}

binary_model.save("binary_model_final.keras")
multiclass_model.save("multiclass_model_final.keras")

with open("cascade_system.json", "w", encoding="utf-8") as f:
    json.dump(cascade_config, f, indent=4)
print("Sistem ve modeller kaydedildi")


