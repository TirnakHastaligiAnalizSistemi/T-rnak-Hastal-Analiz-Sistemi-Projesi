# -*- coding: utf-8 -*-
import streamlit as st
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Başlık
st.set_page_config(
    page_title="Nail Disease Detection",
    page_icon="🧬",
    layout="centered"
)

st.title("🧬 Tırnak Hastalığı Analiz Sistemi")
st.write("DenseNet121 tabanlı: Healthy vs Disease + Hastalık Tipi + Sistemik Risk Analizi")

# Ana Modellerin Çağrılması
MODEL_PATH = r"C:\Users\metin\PyCharmMiscProject\ikili_sistem.keras"

# Sınıf isimlerinin okunması için klasör eklenmesi
TRAIN_DIR = r"C:\Users\metin\OneDrive - CAKABEY OKULLARI\Masaüstü\data\data\train"

model = tf.keras.models.load_model(MODEL_PATH)

# Klasörden isim okuma
raw_classes = [
    d for d in os.listdir(TRAIN_DIR)
    if os.path.isdir(os.path.join(TRAIN_DIR, d))
]

CLASS_NAMES = sorted(raw_classes)
CLASS_NAMES_LOWER = [c.lower() for c in CLASS_NAMES]

if "healthy" not in CLASS_NAMES_LOWER:
    st.error("❌ TRAIN_DIR içinde 'healthy' sınıfı bulunamadı. Klasör isimlerini kontrol et.")
    st.stop()

HEALTHY_INDEX = CLASS_NAMES_LOWER.index("healthy")

# Sistemik risk tabloları
SYSTEMIC_RISKS = {
    "psoriasis": {
        "Psoriatik artrit": 0.40,
        "Psoriasis vulgaris": 0.65,
        "Metabolik sendrom": 0.15,
        "Kardiyovasküler risk": 0.10
    },

    "acral_lentiginouns_melanoma": {
        "ALM tırnak tutulumu": 0.25,
        "ALM etnik prevalans": 0.30
    },

    "onychomycosis": {
        "Diyabet": 0.25,
        "Damar hastalığı": 0.15,
        "İleri yaş": 0.35,
        "İmmün yetmezlik": 0.07
    },

    "clubbing": {
        "Akciğer hastalığı": 0.50,
        "Kardiyovasküler": 0.15,
        "Karaciğer/GİS": 0.25,
        "Endokrin": 0.10
    },

    "blue_finger": {
        "Periferik siyanoz": 0.45,
        "Kardiyak hastalık": 0.12,
        "Pulmoner hastalık": 0.12,
        "Böbrek/hematolojik": 0.07,
        "Travma": 0.28
    },
    "pitting": {
        "sedef": 0.75,
        "Saçkıran": 0.15,
        "Egzama / Atopik dermatit": 0.15,
        "Reiter sendromu / Psoriatik artrit": 0.10
    }
}

# Tıbbi Açıklamalar
EXPLANATIONS = {
    "psoriasis": """
Tırnak lezyonları sedef hastalığı olan hastaların yaklaşık yarısında görülür ve yaşam boyu görülme sıklığı %80-90 civarındadır.
Sedef tırnak bulguları: çukurlaşma, yağ damlası görünümü, onikoliz ve subungual keratin birikimi.
(Reich & Szepietowski, 2011)
""",
    "acral_lentiginous_melanoma": """
Acral Lentiginous Melanoma, avuç içi, ayak tabanı ve tırnak yatağı gibi bölgelerde görülen bir melanom türüdür. 
Melanomların %1–3’ünü oluşturur ancak tırnak altı tümörlerde önemlidir.
""",
    "onychomycosis": """
Onychomycosis (tırnak mantarı) diyabet, ileri yaş, periferik damar hastalığı ve immün yetmezlik ile ilişkilidir.
""",
    "clubbing": """
Clubbing (çomak parmak), akciğer hastalıkları, kalp-damar hastalıkları ve karaciğer hastalıklarının önemli bir belirtisi olabilir.
""",

    "blue_finger": """
Mavi tırnak (siyanoz), dolaşım bozukluğu, kalp hastalığı, pulmoner hastalık veya travma kaynaklı olabilir.
"""
}

# Fotoğraf yükleme
def load_and_prepare(img_bytes):
    # DenseNet121 input: 224x224, 0-1 normalizasyon uygun (preprocess_input istersen ekleriz)
    img = image.load_img(img_bytes, target_size=(224, 224))
    img_arr = image.img_to_array(img) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)
    return img_arr

# Hastalıklı sağlıklı karar için gereken oranın belirlenmesi
def predict_pipeline(img_arr, healthy_threshold=0.50):
    preds = model.predict(img_arr)[0]

# Tüm sınıfların olasılıkları
    class_probs = {CLASS_NAMES_LOWER[i]: float(preds[i]) for i in range(len(CLASS_NAMES_LOWER))}

# Sağlıklı Olma olasılıkları
    healthy_prob = class_probs["healthy"]
    harmful_prob = 1.0 - healthy_prob

# En yüksek olasılıklı sınıfın belirlenmesi
    best_idx = int(np.argmax(preds))
    best_class = CLASS_NAMES_LOWER[best_idx]
    best_prob = float(preds[best_idx])

    # Tırnakların Hastalıklı Sağlıklı Olarak Belirlenmesi
    if healthy_prob >= healthy_threshold:
        return {
            "status": "Healthy",
            "healthy_probability": healthy_prob,
            "harmful_probability": harmful_prob,
            "detailed_class": "healthy",
            "detailed_prob": healthy_prob,
            "systemic": None,
            "class_probs": class_probs
        }
    if best_class == "healthy":
        non_healthy_indices = [i for i, c in enumerate(CLASS_NAMES_LOWER) if c != "healthy"]
        non_healthy_probs = preds[non_healthy_indices]
        best_nonhealthy_idx = int(non_healthy_indices[int(np.argmax(non_healthy_probs))])
        best_class = CLASS_NAMES_LOWER[best_nonhealthy_idx]
        best_prob = float(preds[best_nonhealthy_idx])

    # Sistemik Risk Hesaplama
    systemic_map = SYSTEMIC_RISKS.get(best_class, {})
    systemic_results = {name: (best_prob * ratio) for name, ratio in systemic_map.items()}
    return {
        "status": "Harmful",
        "healthy_probability": healthy_prob,
        "harmful_probability": harmful_prob,
        "detailed_class": best_class,
        "detailed_prob": best_prob,
        "systemic": systemic_results,
        "class_probs": class_probs
    }

# Arayüz Tasarımı
uploaded = st.file_uploader("Bir tırnak fotoğrafı yükleyin", type=["jpg", "jpeg", "png"])
healthy_threshold = st.slider(
    "Sağlıklı kabul eşiği (Sağlıklı olma olasılığı ≥ bu değer ise 'sağlıklı' diyecek)",
    min_value=0.30,
    max_value=0.90,
    value=0.50,
    step=0.05
)

if uploaded:
    st.image(uploaded, caption="Yüklenen Görüntü", use_container_width=True)
    img_arr = load_and_prepare(uploaded)

    st.write("### 🔍 Analiz ediliyor...")
    result = predict_pipeline(img_arr, healthy_threshold=healthy_threshold)

    st.write(f"###  Sağlıklı Olasılığı: **{result['healthy_probability']:.2%}**")
    st.write(f"### 🧪 Zararlı Olasılığı: **{result['harmful_probability']:.2%}**")

    if result["status"] == "Healthy":
        st.success("Tırnak genel olarak sağlıklı görünüyor.")
    else:
        st.error("⚠ Tırnakta hastalık belirtisi olabilir!")

        disease = result["detailed_class"]
        st.write(f"### 🎯 Tespit Edilen Hastalık: **{disease.capitalize()}**")
        st.write(f"Model Olasılığı (bu hastalık için): **{result['detailed_prob']:.2%}**")

        # Açıklama
        st.write("### 📘 Tıbbi Açıklama")
        st.write(EXPLANATIONS.get(disease, "Açıklama bulunamadı."))

        # Sistemik risk grafiği
        if result["systemic"]:
            st.write("### 📊 Sistemik Hastalık Risk Dağılımı")

            labels = list(result["systemic"].keys())
            values = list(result["systemic"].values())

            fig, ax = plt.subplots()
            ax.pie(values, labels=labels, autopct="%1.1f%%")
            ax.axis("equal")
            st.pyplot(fig)
        else:
            st.info("Bu hastalık için sistemik risk tablosu tanımlı değil.")

    # Kişinin Tüm Sınıfların Olasılığını Görmesi İçin Tablo
    with st.expander("🔎 Tüm sınıf olasılıklarını göster"):
        probs_table = {
            "Sınıf": [name for name in CLASS_NAMES_LOWER],
            "Olasılık": [result["class_probs"][name] for name in CLASS_NAMES_LOWER]
        }
        st.table(probs_table)
