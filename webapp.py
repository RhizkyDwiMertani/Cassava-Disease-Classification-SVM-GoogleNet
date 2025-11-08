import streamlit as st
import torch
import joblib
import os
import cv2
import numpy as np
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew
from skimage.feature import local_binary_pattern

st.set_page_config(page_title="Deteksi Penyakit Daun Singkong", layout="centered")

# Paths
train_path = "D:/KULIAH IT/Skripsi/uji/train"
test_path = "D:/KULIAH IT/Skripsi/uji/test"
class_names = joblib.load("D:/KULIAH IT/Skripsi/CODE/class_names.pkl")

# Load Models
model_pretrain = models.googlenet(pretrained=True)
model_pretrain.fc = torch.nn.Linear(1024, len(class_names))
model_pretrain.load_state_dict(torch.load("D:/KULIAH IT/Skripsi/CODE/googlenet_pretrain_model.pth", map_location=torch.device('cpu')))
model_pretrain.eval()

model_nopre = models.googlenet(pretrained=False, aux_logits=False)
model_nopre.fc = torch.nn.Linear(1024, len(class_names))
model_nopre.load_state_dict(torch.load("D:/KULIAH IT/Skripsi/CODE/googlenet_model.pth", map_location=torch.device('cpu')))
model_nopre.eval()

svm_model = joblib.load("D:/KULIAH IT/Skripsi/CODE/svm_model.pkl")
pca = joblib.load("D:/KULIAH IT/Skripsi/CODE/pca.pkl")
scaler = joblib.load("D:/KULIAH IT/Skripsi/CODE/scaler.pkl")

# Penyakit Info
disease_info = {
    "bacterial blight": {
        "penyebab": "Disebabkan oleh bakteri *Xanthomonas campestris pv. manihotis*.",
        "solusi": "Gunakan varietas tahan, rotasi tanaman."
    },
    "brown spot": {
        "penyebab": "Jamur *Cercosporidium henningsii*, muncul saat musim hujan.",
        "solusi": "Gunakan fungisida dan varietas tahan."
    },
    "green mite": {
        "penyebab": "Tungau merah, umum saat kemarau.",
        "solusi": "Gunakan musuh alami dan air semprot."
    },
    "mosaic": {
        "penyebab": "Virus mosaik dari kutu kebul.",
        "solusi": "Gunakan tanaman sehat, kendalikan vektor."
    },
    "healthy": {
        "penyebab": "Tanaman dalam kondisi sehat.",
        "solusi": "Lakukan monitoring dan budidaya baik."
    }
}

# Preprocessing
def preprocess_image(img):
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    return transform(img).unsqueeze(0)

def predict_googlenet_pretrain(img):
    tensor = preprocess_image(img)
    output = model_pretrain(tensor)
    _, pred = torch.max(output, 1)
    return class_names[pred.item()]

def predict_googlenet_nopre(img):
    tensor = preprocess_image(img)
    output = model_nopre(tensor)
    _, pred = torch.max(output, 1)
    return class_names[pred.item()]

def extract_features_svm(img):
    img = cv2.resize(img, (224, 224))
    hist = cv2.calcHist([img], [0, 1, 2], None, [8]*3, [0, 256]*3)
    hist = cv2.normalize(hist, hist).flatten()

    moments = []
    for i in range(3):
        ch = img[:, :, i]
        moments.extend([np.mean(ch), np.std(ch), skew(ch.flatten())])
    moments = np.array(moments)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, 8, 1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10))
    lbp_hist = lbp_hist.astype("float") / (lbp_hist.sum() + 1e-6)

    return np.concatenate([hist, moments, lbp_hist]).reshape(1, -1)

def predict_svm(img):
    features = extract_features_svm(img)
    scaled = scaler.transform(features)
    reduced = pca.transform(scaled)
    pred = svm_model.predict(reduced)
    return class_names[pred[0]]

# Evaluasi dan Cache
@st.cache_resource
def evaluate_models():
    cache_path = "D:/KULIAH IT/Skripsi/CODE/eval_results.pkl"
    if os.path.exists(cache_path):
        return joblib.load(cache_path)

    true_labels, pred_pre, pred_nopre, pred_svm = [], [], [], []

    for label in class_names:
        folder = os.path.join(test_path, label)
        if not os.path.isdir(folder): continue
        for fname in os.listdir(folder):
            try:
                path = os.path.join(folder, fname)
                img = Image.open(path).convert("RGB")
                img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                true_labels.append(label)
                pred_pre.append(predict_googlenet_pretrain(img))
                pred_nopre.append(predict_googlenet_nopre(img))
                pred_svm.append(predict_svm(img_cv))
            except:
                continue

    acc_pre = accuracy_score(true_labels, pred_pre)
    acc_nopre = accuracy_score(true_labels, pred_nopre)
    acc_svm = accuracy_score(true_labels, pred_svm)
    cm_pre = confusion_matrix(true_labels, pred_pre, labels=class_names)
    cm_nopre = confusion_matrix(true_labels, pred_nopre, labels=class_names)
    cm_svm = confusion_matrix(true_labels, pred_svm, labels=class_names)

    result = (acc_pre, acc_nopre, acc_svm, cm_pre, cm_nopre, cm_svm)
    joblib.dump(result, cache_path)
    return result

# UI
st.title("🌿 Deteksi Penyakit Daun Singkong")
st.caption("Prediksi otomatis menggunakan SVM dan GoogLeNet (Pretrained & Non-Pretrained)")

# Input
image = None
input_opt = st.radio("📤 Pilih metode input:", ["Unggah Gambar", "Gunakan Kamera"])
if input_opt == "Unggah Gambar":
    file = st.file_uploader("Unggah gambar daun", type=["jpg", "png"])
    if file: image = Image.open(file).convert("RGB")
elif input_opt == "Gunakan Kamera":
    capture = st.camera_input("Ambil gambar")
    if capture: image = Image.open(capture).convert("RGB")

if image:
    st.image(image, caption="Gambar Input", use_column_width=True)
    pred_svm = predict_svm(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
    pred_nopre = predict_googlenet_nopre(image)
    pred_pre = predict_googlenet_pretrain(image)

    st.markdown("### 🧠 Hasil Deteksi Model")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**📘 SVM**")
        st.info(f"Hasil: **{pred_svm.upper()}**")
        st.markdown(f"**Penyebab:** {disease_info[pred_svm]['penyebab']}")
        st.markdown(f"**Solusi:** {disease_info[pred_svm]['solusi']}")

    with col2:
        st.markdown("**🧪 GoogLeNet Tanpa Pretrain**")
        st.warning(f"Hasil: **{pred_nopre.upper()}**")
        st.markdown(f"**Penyebab:** {disease_info[pred_nopre]['penyebab']}")
        st.markdown(f"**Solusi:** {disease_info[pred_nopre]['solusi']}")

    with col3:
        st.markdown("**🤖 GoogLeNet Pretrained**")
        st.success(f"Hasil: **{pred_pre.upper()}**")
        st.markdown(f"**Penyebab:** {disease_info[pred_pre]['penyebab']}")
        st.markdown(f"**Solusi:** {disease_info[pred_pre]['solusi']}")

# Evaluasi
st.markdown("---")
acc_pre, acc_nopre, acc_svm, cm_pre, cm_nopre, cm_svm = evaluate_models()
st.markdown("### 📊 **Ringkasan Akurasi & Confusion Matrix**")

col1, col2, col3 = st.columns([1, 1, 1], gap="large")

with col1:
    st.markdown("""
        <div style='font-size:16px; text-align:center;'>
            <strong style='font-size:18px;'>📘 SVM</strong><br>
            Akurasi: <span style='color:limegreen; font-weight:bold;'>{:.2%}</span>
        </div>
    """.format(acc_svm), unsafe_allow_html=True)
    fig = plt.figure(figsize=(5, 4))
    sns.heatmap(cm_svm, annot=True, xticklabels=class_names, yticklabels=class_names, fmt="d", cmap="Blues")
    plt.xlabel("Prediksi")
    plt.ylabel("Aktual")
    plt.title("Confusion Matrix SVM", fontsize=10)
    st.pyplot(fig)

with col2:
    st.markdown("""
        <div style='font-size:16px; text-align:center;'>
            <strong style='font-size:18px;'>🧪 GoogLeNet Tanpa Pretrain</strong><br>
            Akurasi: <span style='color:orange; font-weight:bold;'>{:.2%}</span>
        </div>
    """.format(acc_nopre), unsafe_allow_html=True)
    fig = plt.figure(figsize=(5, 4))
    sns.heatmap(cm_nopre, annot=True, xticklabels=class_names, yticklabels=class_names, fmt="d", cmap="Oranges")
    plt.xlabel("Prediksi")
    plt.ylabel("Aktual")
    plt.title("Confusion Matrix GoogLeNet (No Pretrain)", fontsize=10)
    st.pyplot(fig)

with col3:
    st.markdown("""
        <div style='font-size:16px; text-align:center;'>
            <strong style='font-size:18px;'>🤖 GoogLeNet Pretrained</strong><br>
            Akurasi: <span style='color:limegreen; font-weight:bold;'>{:.2%}</span>
        </div>
    """.format(acc_pre), unsafe_allow_html=True)
    fig = plt.figure(figsize=(5, 4))
    sns.heatmap(cm_pre, annot=True, xticklabels=class_names, yticklabels=class_names, fmt="d", cmap="Greens")
    plt.xlabel("Prediksi")
    plt.ylabel("Aktual")
    plt.title("Confusion Matrix GoogLeNet (Pretrained)", fontsize=10)
    st.pyplot(fig)


#streamlit run "D:/KULIAH IT/Skripsi/CODE/webapp.py"
