🧠 Breast Cancer Classification System

This project is a Flask-based web application that uses deep learning models to classify breast cancer images into:

Benign
Malignant
Normal

It also provides Grad-CAM visualizations to highlight important regions used by the model for prediction.

🚀 Features
🔍 Image classification using:
    ResNet101 (with Grad-CAM)
    Vision Transformer (ViT)
    🖼️ Upload single or multiple images
    🔥 Visual explanation using Grad-CAM
    📊 Confidence scores & class probabilities
    🧹 Clear uploaded files functionality
    🛠️ Tech Stack
    Backend: Flask
    Deep Learning: PyTorch, Torchvision
    Visualization: Grad-CAM
    Image Processing: PIL, OpenCV
    Model: ResNet101, ViT
    📁 Project Structure
Breast_Cancer_Detector/
│
├── app.py
├── models/
│   ├── best_resnet101.pth
│   └── best_vit.pth
├── static/
│   └── uploads/
├── templates/
│   └── index.html
├── requirements.txt
└── README.md
▶️ How to Run the Project
1️⃣ Clone the repository
git clone https://github.com/MahithaVedampudi/BreastCancerClassification.git
cd BreastCancerClassification
2️⃣ Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
3️⃣ Install dependencies
pip install -r requirements.txt
4️⃣ Add model files

Place trained models inside the models/ folder:

models/
├── best_resnet101.pth
└── best_vit.pth
5️⃣ Run the app
python app.py

Open browser:

http://127.0.0.1:5000/
📊 Dataset Information

⚠️ Dataset is NOT included in this repository

❓ Why is the dataset missing?
GitHub has a file size limit (100MB per file)
Datasets are usually very large (hundreds of MBs or GBs)
Uploading them makes the repo heavy and slow
📥 Where to get the dataset?

You can use publicly available datasets like:

Breast Cancer Histopathological Images (BreakHis)
Kaggle Breast Cancer datasets
https://www.kaggle.com/datasets

Search:

breast cancer histopathology images
🧠 Model Information
📌 ResNet101
Used for:
Prediction
Grad-CAM visualization
Final layer modified for 3-class classification
📌 Vision Transformer (ViT)
Used only for:
Prediction (no Grad-CAM)
Loaded using timm
💾 How Models Are Stored
Models are saved as .pth files:
best_resnet101.pth
best_vit.pth
These contain:
Model weights (state_dict)
Learned parameters after training
❓ Why models are not on GitHub sometimes?
Large file size
GitHub limitations
Best practice:
Upload via:
Google Drive
Kaggle
GitHub Releases
🖼️ How Files Are Handled
Upload Flow:
User uploads image
Image saved in:
static/uploads/
Processing:
Image → Tensor
Passed to models
Prediction generated
Output:
Prediction JSON
Grad-CAM image saved
🧹 Clear Function
/clear route deletes all uploaded files:
shutil.rmtree(UPLOAD_FOLDER)
📌 API Endpoints
/upload
Method: POST
Input: Images (+ optional masks)
Output:
{
  "class": "benign",
  "confidence": 0.92
}
/clear
Clears uploaded files
⚠️ Notes
If models are missing:
WARNING: models not found
If both models are missing:
ERROR: No models available!
🔮 Future Improvements
Add dataset download automation
Deploy using Docker / Cloud
Add more models (EfficientNet, DenseNet)
Improve UI/UX
