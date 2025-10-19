# 🧠 Brain Tumor Detection Using VGG16

## 📘 Overview
This project focuses on **brain tumor detection from MRI images** using **transfer learning with the VGG16 convolutional neural network (CNN)**. The goal is to accurately classify MRI scans into one of three tumor categories or as “no tumor” by fine-tuning a pre-trained model for this medical imaging task.

---

## ⚙️ Model Architecture and Methodology

### 🔹 Transfer Learning with VGG16
We utilize **VGG16**, a pre-trained CNN trained on the **ImageNet dataset (1.4 million images)**, as our feature extractor.

- **Model Initialization:**
  ```python
  base_model = VGG16(input_shape=(128,128,3), include_top=False, weights='imagenet')
  ```
  - `input_shape=(128,128,3)` matches the MRI image dimensions.
  - `include_top=False` removes VGG16’s original classification head.
  - `weights='imagenet'` loads pre-trained ImageNet weights for rich feature extraction.

- **Freezing Layers:**
  Initially, all layers of the base model are frozen to preserve the learned ImageNet weights:
  ```python
  for layer in base_model.layers:
      layer.trainable = False
  ```

- **Fine-Tuning:**
  To improve domain-specific learning, the last three convolutional layers are unfrozen:
  ```python
  base_model.layers[-2].trainable = True
  base_model.layers[-3].trainable = True
  base_model.layers[-4].trainable = True
  ```

- **Model Construction:**
  The new classification head is added on top of VGG16:
  ```python
  model = Sequential()
  model.add(base_model)
  model.add(Flatten())
  model.add(Dropout(0.3))
  model.add(Dense(128, activation='relu'))
  model.add(Dropout(0.2))
  model.add(Dense(len(unique_labels), activation='softmax'))
  ```
  - `Flatten()` reshapes the convolutional output.
  - `Dropout()` layers reduce overfitting.
  - The final dense layer uses **softmax** activation for multi-class classification.

---

## 🧩 Dataset Description
The dataset consists of **MRI scans categorized into four tumor classes and one non-tumor category**, divided into separate training and testing folders.

**Dataset structure:**
```
Training/
  ├── glioma/
  ├── meningioma/
  ├── notumor/
  └── pituitary/

Testing/
  ├── glioma/
  ├── meningioma/
  ├── notumor/
  └── pituitary/
```
Each subfolder contains MRI images corresponding to its respective class. The dataset is automatically downloaded using `gdown` from a Google Drive source.

---

## 🧠 Training Configuration
- **Optimizer:** `Adam(learning_rate=0.0001)`
- **Loss Function:** `sparse_categorical_crossentropy`
- **Metrics:** `sparse_categorical_accuracy`
- **Label Encoding:** Integer-encoded class labels
- **Batch Size:** Defined within the data generator
- **Epochs:** Typically between 10–30

```python
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])
```

---

## 📊 Evaluation Metrics
The trained model’s performance is assessed using multiple evaluation techniques:

- ✅ **Sparse Categorical Accuracy** – overall model correctness
- ✅ **Confusion Matrix** – shows per-class prediction distribution
- ✅ **Classification Report** – precision, recall and F1-score for each class
- ✅ **ROC Curve / AUC** – measures discriminative power of the model


---

## 🧰 Tools and Libraries
- **TensorFlow / Keras** – deep learning framework
- **OpenCV, Pillow** – image preprocessing
- **NumPy, Matplotlib, Seaborn** – numerical computation and visualization
- **Scikit-learn** – evaluation and performance metrics
- **gdown** – for automated dataset download

---

## 🚀 Results Summary
The fine-tuned VGG16 model achieves **high classification accuracy** and strong **AUC scores** across all classes. ROC and confusion matrix visualizations show clear class separations with minimal misclassifications.


---

## 🔮 Future Enhancements
- Implement **Grad-CAM** for explainable AI visualization.
- Deploy the model using **Flask** or **Streamlit** for real-time predictions.
- Extend the dataset to include **3D MRI volumes** for enhanced medical insight.

---

## 👨‍💻 Author
**Manpreet Singh**  
Brain Tumor Detection using Deep Learning – 2025

