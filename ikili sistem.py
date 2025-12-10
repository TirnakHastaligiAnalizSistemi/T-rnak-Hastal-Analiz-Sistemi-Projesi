# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.densenet import preprocess_input
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# DenseNet modellerinin altılı ve ikili sınıflandırmadaki sonuçlarının geri çağırılması
binary_model_path = r"best_densenet121binary.keras"
multiclass_model_path = r"best_densenet121.keras"

binary_model = tf.keras.models.load_model(binary_model_path)
multiclass_model = tf.keras.models.load_model(multiclass_model_path)

print("Modeller başarıyla yüklendi.")

# Modeli Kaydetme
class CascadeNailModel(tf.keras.Model):
    def __init__(self, binary_model, multiclass_model, threshold=0.63):
        super().__init__()
        self.binary_model = binary_model
        self.multiclass_model = multiclass_model
        self.threshold = threshold
    def call(self, inputs, training=False):
        binary_probs = self.binary_model(inputs, training=False)
        harmful_prob = binary_probs[:, 1]
        mask = harmful_prob >= self.threshold
        multiclass_probs = self.multiclass_model(inputs, training=False)
        predicted_classes = tf.argmax(multiclass_probs, axis=1)
        output = tf.where(
            mask,
            predicted_classes,
            tf.constant(-1, dtype=tf.int64)
        )
        return output

# Kaydetmek için cascate modeli oluşturmak
cascade_model = CascadeNailModel(
    binary_model=binary_model,
    multiclass_model=multiclass_model,
    threshold=0.63
)
dummy_input = tf.zeros((1, 224, 224, 3))
cascade_model(dummy_input)

# Sistemi kaydetme
cascade_model.save("ikili_sistem.keras")
print("model kaydedildi")

# Veri klasörlerinin yüklenmesi
train_dir = r"C:\Users\metin\OneDrive - CAKABEY OKULLARI\Masaüstü\data\data\train"
train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=preprocess_input
).flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

class_names = list(train_gen.class_indices.keys())
print("\nSınıflar:", class_names)

# İkili model ile altılı modelin birleştirip eğitiminin yapılması

y_true = []
y_pred = []

binary_threshold = 0.63

for i in range(len(train_gen)):
    img, label = train_gen[i]
    y_true.append(np.argmax(label))

    # İkili sınıflandırmadaki (hastalıklı sağlıklı) gelecek sonucun hesaplanması
    binary_prob = binary_model.predict(img, verbose=0)[0]
    harmful_prob = binary_prob[1]

    # Zararlıysa 6'lı sınıflandırmaya gönderilmesi
    if harmful_prob < binary_threshold:
        y_pred.append(-1)
        continue

    # 6'lı sınıflandırmada tahminler
    multiclass_prob = multiclass_model.predict(img, verbose=0)[0]
    predicted_class = np.argmax(multiclass_prob)
    y_pred.append(predicted_class)

print("\nTahminler tamamlandı.")

# Karışıklık Matrisi
valid_indices = [i for i in range(len(y_pred)) if y_pred[i] != -1]
y_true_filtered = [y_true[i] for i in valid_indices]
y_pred_filtered = [y_pred[i] for i in valid_indices]

print("\nİkili Sistemde hesaplama için kullanılan görüntü sayısı:", len(y_true_filtered))
print("\nSınıflandırma Raporu:\n")
print(classification_report(y_true_filtered, y_pred_filtered, target_names=class_names))
cm = confusion_matrix(y_true_filtered, y_pred_filtered)
print(cm)
#Karışıklılık Matrisinin Grafiğinin Yazdırılması
plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(xticks_rotation=45)
plt.title("İkili Sistemde Doğruluk")
plt.tight_layout()
plt.show()

# Sistemin Doğruluk Oranı
accuracy = np.mean(np.array(y_true_filtered) == np.array(y_pred_filtered))
print(f"\nİkili Sistemde Doğruluk: %{accuracy*100:.2f}")
