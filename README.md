# 🧠 Breast Cancer Classification System

A **Flask-based AI web application** that classifies breast cancer histopathology images into:

- ✅ Benign  
- ⚠️ Malignant  
- 🟢 Normal  

It also provides **Grad-CAM visualizations** to highlight regions influencing the model’s decision.

---

## 🚀 Features

- 🔍 Dual-model prediction:
  - **ResNet101** (with Grad-CAM visualization)
  - **Vision Transformer (ViT)**
- 🖼️ Upload single or multiple images
- 🔥 Visual explanations (Grad-CAM)
- 📊 Confidence scores & probability distribution
- 🧹 Clear uploaded images functionality

---

## 🛠️ Tech Stack

| Category        | Tools Used                          |
|----------------|------------------------------------|
| Backend        | Flask                              |
| Deep Learning  | PyTorch, Torchvision               |
| Visualization  | Grad-CAM                           |
| Image Handling | PIL, OpenCV                        |
| Models         | ResNet101, Vision Transformer (ViT)|



---

## ▶️ How to Run the Project

### 1️⃣ Clone the Repository

bash
git clone https://github.com/MahithaVedampudi/BreastCancerClassification.git
cd BreastCancerClassification
2️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate   # Windows

## 📁 Project Structure

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Add Model Files

Place trained models inside the models/ folder:
models/
├── best_resnet101.pth
└── best_vit.pth

5️⃣ Run the Application
python app.py
Open in browser:

http://127.0.0.1:5000/

📊 Dataset Information

⚠️ Dataset is NOT included in this repository

❓ Why is the dataset missing?
GitHub has a 100MB file size limit
Datasets are typically large (hundreds of MBs/GBs)
Keeping datasets out improves repo performance
📥 Where to Download Dataset?

You can use:

Breast Cancer Histopathological Images (BreakHis)
Kaggle datasets 👉 https://www.kaggle.com/datasets

Search for:

breast cancer histopathology images
🧠 Model Details
🔹 ResNet101
Used for prediction + Grad-CAM
Final layer modified for 3-class classification
🔹 Vision Transformer (ViT)
Used for prediction only
Loaded using timm
💾 Model Storage

Models are stored as .pth files:

best_resnet101.pth
best_vit.pth

These contain:

Trained weights (state_dict)
Learned parameters
📦 Why Models May Be Missing?
Large file size
GitHub limitations

👉 Recommended alternatives:

Google Drive
Kaggle
GitHub Releases
🖼️ File Handling Workflow
Upload image via UI
Saved in:
static/uploads/
Processing:
Image → Tensor
Model prediction
Grad-CAM generation
Output:
Prediction JSON
Heatmap image
🧹 Clear Uploaded Files

Endpoint:

POST /clear

Deletes all uploaded files using:

shutil.rmtree(UPLOAD_FOLDER)
📌 API Endpoints
🔹 /upload (POST)

Input:

Images (+ optional masks)

Output:

{
  "class": "benign",
  "confidence": 0.92
}
🔹 /clear (POST)
Clears uploaded images
⚠️ Notes

If models are missing:

WARNING: models not found

If both models are unavailable:

ERROR: No models available!
🔮 Future Improvements
🚀 Add dataset auto-download
☁️ Cloud deployment (AWS / Render)
🧠 Add more models (EfficientNet, DenseNet)
🎨 Improve UI/UX
