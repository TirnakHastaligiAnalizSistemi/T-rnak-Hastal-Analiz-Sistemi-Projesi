import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.python.layers.core import Dropout
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# Klasörler
train_dir = r"C:\Users\metin\OneDrive - CAKABEY OKULLARI\Masaüstü\data\data\train"
test_dir = r"C:\Users\metin\OneDrive - CAKABEY OKULLARI\Masaüstü\data\data\test"
validation_dir = r"C:\Users\metin\OneDrive - CAKABEY OKULLARI\Masaüstü\data\data\validation"

# Parametreler
img_size = (224, 224)
batch_size = 32
epochs = 25
seed = 42
Dropout(0.5)

# Verileri Yükleme
train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)
train_ds = train_gen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    seed=seed
)
val_ds = train_gen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    seed=seed
)
test_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=preprocess_input
).flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

num_classes = train_ds.num_classes

# Model oluşturma
base = VGG16(weights="imagenet", include_top=False, input_shape=(*img_size, 3))
for layer in base.layers:
    layer.trainable = False

x = layers.GlobalAveragePooling2D()(base.output)
x = layers.Dropout(0.3)(x)
x = layers.Dense(256, activation="relu")(x)
out = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs=base.input, outputs=out)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Geri çağırmalar ve kaydetmeler
ckpt = callbacks.ModelCheckpoint(
    "best_vgg16.keras",
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)
early = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True,
    verbose=1
)
reduce = callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=1e-6,
    verbose=1
)


# Model eğitimi
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[ckpt, early, reduce],
    verbose=2
)

# Sonuç

print("\nEn düşük doğrulama kaybı:", min(history.history["val_loss"]))
print("En yüksek doğrulama doğruluğu:", max(history.history["val_accuracy"]))
test_loss, test_acc = model.evaluate(test_gen, verbose=1)
print(f"Test Kayıp: {test_loss:.4f} | Test Doğruluk: {test_acc:.4f}")

# Eğitim-Doğrulama Kaybı Grafiği

plt.figure(figsize=(7, 5))
plt.plot(history.history["loss"], label="Eğitim kaybı")
plt.plot(history.history["val_loss"], label="Doğrulama kaybı")
plt.xlabel("Epoch")
plt.ylabel("Kayıp")
plt.legend()
plt.title("VGG16 Eğitim & Doğrulama Kayıpları")
plt.show()

# Karışıklık Matrisi
class_names = list(test_gen.class_indices.keys())
print("\nSınıf isimleri (index sırası):", class_names)
test_gen.reset()
y_prob = model.predict(test_gen, verbose=1)
y_pred = np.argmax(y_prob, axis=1)
y_true = test_gen.classes
cm = confusion_matrix(y_true, y_pred)
print("\n=== Classification Report ===\n")
print(classification_report(y_true, y_pred, target_names=class_names))

# Karışıklık Matrisinin Grafiğinin Yazdırılması
plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(xticks_rotation=45)
plt.title("VGG16 – Confusion Matrix (Test Set)")
plt.tight_layout()
plt.show()
