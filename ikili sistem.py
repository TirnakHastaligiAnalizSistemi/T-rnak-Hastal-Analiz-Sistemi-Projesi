# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import json
import matplotlib.pyplot as plt
from tensorflow.keras.applications.densenet import preprocess_input
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score
)

# Modeli yüklüyoruz
binary_model_path = "best_densenet121binary.keras"
multiclass_model_path = "best_densenet121.keras"
binary_model = tf.keras.models.load_model(binary_model_path)
multiclass_model = tf.keras.models.load_model(multiclass_model_path)
print("Modeller yüklendi.")
BINARY_THRESHOLD = 0.63

# Kayıtta bir sıkıntı olursa diye sistemimizi pipeline olarak da kaydediyoruz
pipeline_config = {
    "binary_model": binary_model_path,
    "multiclass_model": multiclass_model_path,
    "binary_threshold": BINARY_THRESHOLD,
    "description": "İkili sınıftan altılı sınıfa geçişli ikili sistem"
}
with open("cascade_pipeline_config.json", "w", encoding="utf-8") as f:
    json.dump(pipeline_config, f, indent=4)
print("Pipeline kaydedildi")

# Hem isimlerin alınması hem de işlemlerin gerçekleşmesi için veri seti yüklüyoruz
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
print("Hastalık sınıfları:", class_names)

# Modellerin ikili sistem halini alması
y_true = []
y_pred = []
y_binary_decision = []

for i in range(len(data_gen)):
    img, label = data_gen[i]
    true_class = np.argmax(label)
    y_true.append(true_class)
    binary_probs = binary_model.predict(img, verbose=0)[0]
    harmful_prob = binary_probs[1]
    if harmful_prob < BINARY_THRESHOLD:
        y_pred.append(-1)
        y_binary_decision.append(0)
        continue
    multiclass_probs = multiclass_model.predict(img, verbose=0)[0]
    pred_class = np.argmax(multiclass_probs)
    y_pred.append(pred_class)
    y_binary_decision.append(1)
print("Sistem Tamamlandı")

# Modellerin Kaydedilmesi
binary_model.save("binary_model_final.keras")
multiclass_model.save("multiclass_model_final.keras")
print(" Modeller kaydedildi.")

# Karışıklık Matrisi
valid_idx = [i for i in range(len(y_pred)) if y_pred[i] != -1]
y_true_f = [y_true[i] for i in valid_idx]
y_pred_f = [y_pred[i] for i in valid_idx]

print("\n Karısıklık Matrisi:\n")
print(classification_report(y_true_f, y_pred_f, target_names=class_names))
cm = confusion_matrix(y_true_f, y_pred_f)
plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)
disp.plot(xticks_rotation=45)
plt.title("Sistem Karısıklık Matrisi")
plt.tight_layout()
plt.show()
multiclass_acc = accuracy_score(y_true_f, y_pred_f)
print(f"İkili Sistem 6'lı Sınıflandırmadan Cıkan Dogruluk: %{multiclass_acc*100:.2f}")

# İkili Sistemin Dogrulugunu Buluyoruz
binary_true = [1 if y != -1 else 0 for y in y_true]
binary_pred = y_binary_decision
binary_acc = accuracy_score(binary_true, binary_pred)
print(f"\nİkili Sistem Dogrulugu: %{binary_acc*100:.2f}")
