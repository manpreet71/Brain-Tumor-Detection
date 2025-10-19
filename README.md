# Brain-Tumor-Detection[readme.md](https://github.com/user-attachments/files/22989561/readme.md)
# Brain Tumor Detection using VGG16

## 🧠 Overview
This project implements **brain tumor detection from MRI images** using **transfer learning with the VGG16 CNN architecture**. The model is trained to classify MRI scans as tumor or non-tumor by leveraging a pre-trained VGG16 base model fine-tuned on the custom dataset.

---

## ⚙️ Model Architecture and Methodology

### 🔹 Transfer Learning with VGG16
We use **VGG16**, a pre-trained convolutional neural network trained on the ImageNet dataset (1.4 million images), as the base model.

- **Model Initialization:**
  ```python
  base_model = VGG16(input_shape=(128,128,3), include_top=False, weights='imagenet')
  ```
  - `input_shape=(128,128,3)` matches the MRI image dimensions.
  - `include_top=False` removes the final dense layers of VGG16.
  - `weights='imagenet'` loads pre-trained ImageNet weights for feature extraction.

- **Freezing Layers:**
  All layers of the base model are initially frozen to retain learned ImageNet features:
  ```python
  for layer in base_model.layers:
      layer.trainable = False
  ```

- **Fine-tuning:**
  The last three convolutional layers are made trainable for domain adaptation:
  ```python
  base_model.layers[-2].trainable = True
  base_model.layers[-3].trainable = True
  base_model.layers[-4].trainable = True
  ```

- **Model Construction:**
  The new classification head added on top of VGG16 consists of:
  ```python
  model = Sequential()
  model.add(base_model)
  model.add(Flatten())
  model.add(Dropout(0.3))
  model.add(Dense(128, activation='relu'))
  model.add(Dropout(0.2))
  model.add(Dense(len(unique_labels), activation='softmax'))
  ```
  - `Flatten()` reshapes VGG16 output.
  - `Dropout()` layers help prevent overfitting.
  - The final dense layer uses **softmax** activation for multi-class probability output.

---

## 🧩 Dataset
The model uses the **Brain_Tumor_MRI_Dataset**, which contains two main folders:
```
Training/
  ├── yes/  # Tumor images
  └── no/   # Non-tumor images
Testing/
  ├── yes/
  └── no/
```
The dataset is automatically downloaded from Google Drive using the file ID configured in the notebook.

---

## 🧠 Training Configuration
- **Optimizer:** `Adam(learning_rate=0.0001)`
- **Loss Function:** `categorical_crossentropy`
- **Metrics:** Accuracy
- **Batch Size:** As per `datagen()` configuration
- **Epochs:** Defined in notebook (usually between 10–30 depending on resources)

---

## 📊 Evaluation Metrics
Model performance is evaluated using multiple metrics from scikit-learn:

- ✅ **Accuracy** – overall prediction correctness
- ✅ **Confusion Matrix** – distribution of correct/incorrect predictions
- ✅ **Classification Report** – precision, recall, F1-score
- ✅ **ROC Curve / AUC** – trade-off between TPR and FPR

Example evaluation code:
```python
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
auc = roc_auc_score(y_true, y_pred_probs)
plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
plt.legend()
plt.title('ROC Curve')
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
print(classification_report(y_true, y_pred_classes))
```

---

## 🧠 Tools and Libraries
- **TensorFlow / Keras** – deep learning framework
- **OpenCV, Pillow** – image preprocessing
- **NumPy, Matplotlib, Seaborn** – numerical operations and visualizations
- **Scikit-learn** – evaluation metrics
- **gdown** – dataset download from Google Drive

---

## 🚀 Results
The fine-tuned VGG16 model achieves strong classification accuracy and AUC scores on test MRI images. The ROC and confusion matrix plots indicate robust tumor vs. non-tumor separation.

*(You can insert accuracy plots, confusion matrix, and ROC curve images in an `/assets` folder.)*

---

## 🧾 Future Enhancements
- Introduce **Grad-CAM** visualization for model explainability
- Deploy model via **Flask** or **Streamlit** web app
- Extend to multi-class tumor type detection

---

## 👨‍💻 Author
**Manpreet Singh**  
Brain Tumor Detection using Deep Learning – 2025

