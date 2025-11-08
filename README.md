# 🌿 Cassava Leaf Disease Classification System (SVM vs CNN – GoogLeNet)

## 📘 Overview
Cassava (*Manihot esculenta*) is one of the main food crops that is highly susceptible to various leaf diseases, which can significantly reduce crop productivity.  
This project aims to **compare the efficiency of two image classification methods** — **Support Vector Machine (SVM)** and **Convolutional Neural Network (CNN)** with the **GoogLeNet architecture** — in classifying cassava leaf diseases.  

The system is designed to provide **instant disease diagnosis through a web-based platform**, making it accessible for both farmers and agricultural researchers.

---

## 🎯 Objectives
- Compare the performance of **SVM** and **CNN (GoogLeNet)** in cassava leaf disease classification.  
- Develop an **accessible web-based diagnosis system** for real-time detection.  
- Evaluate the impact of **pretrained vs non-pretrained** CNN models on classification accuracy.  

---

## 🧠 Methodology
1. **Dataset Preparation**  
   - Combination of **primary data** (captured from the field) and **secondary data** (publicly available images).  

2. **Data Preprocessing & Augmentation**  
   - Image resizing, normalization, and data augmentation to improve model robustness.  

3. **Feature Extraction & Classification**  
   - **SVM:** Utilized handcrafted features such as color, texture, and statistical descriptors.  
   - **CNN (GoogLeNet):** Implemented deep learning architecture with and without pretrained weights.  

4. **Model Evaluation**  
   - Compared model accuracy and efficiency using validation metrics.  

---

## 📊 Results
| Model                         | Pretraining | Accuracy |
|-------------------------------|--------------|-----------|
| CNN – GoogLeNet               | ✅ Yes        | **95%**   |
| CNN – GoogLeNet               | ❌ No         | **89%**   |
| Support Vector Machine (SVM)  | N/A          | **83%**   |

> The results show that pretrained CNN models outperform traditional SVM and non-pretrained CNNs in image classification tasks.

---

## 🛠️ Tools & Technologies
- **Programming Language:** Python  
- **Libraries:** TensorFlow, Keras, Scikit-learn, OpenCV, NumPy, Pandas, Matplotlib  
- **Development Environment:** Google Colab, Jupyter Notebook  
- **Deployment (optional):** Flask / Streamlit  

---

## 🚀 Key Features
- Automated cassava leaf disease detection system  
- Web-based platform for easy accessibility  
- Comparison between traditional ML and deep learning approaches  
- High accuracy and reliable classification results  

---

## 📈 Future Improvements
- Integrate the system into a **mobile application** for field use.  
- Expand dataset with more diverse environmental conditions.  
- Optimize model size and speed for real-time deployment.  

---

## 👩‍💻 Author
**Rhizky Dwi Mertani**  
Bachelor of Informatics, Ahmad Dahlan University  
Passionate about Data Science, Machine Learning, and AI-driven solutions.  

---

⭐ *If you found this project interesting, consider giving it a star!*
