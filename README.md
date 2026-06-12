# 🔍 Deepfake Detection App – AI-Powered Video Forensics

Web application for deepfake video detection using a hybrid spatial‑frequency architecture with adaptive fusion and Domain‑Adversarial Neural Network (DANN). Developed as part of a Master’s thesis in Cybersecurity.

## ✨ Features

- 🎥 Upload video (MP4, AVI, MOV, etc.)
- 🧠 Frame‑by‑frame face detection & alignment (SSD + MTCNN)
- 🌐 Dual‑branch analysis: **spatial** (EfficientNet‑B0) and **frequency** (FFT)
- ⚖️ Attention‑based adaptive fusion
- 📊 Temporal aggregation (12 features + logistic regression)
- 📈 Per‑frame probabilities, temporal evolution, face gallery
- ✅ Final verdict: **REAL** / **DEEPFAKE** with confidence score

## 🛠️ Requirements

- Python 3.10 or 3.11
- pip + virtualenv (recommended)

## 📦 Installation

```bash
git clone <your-repo-url>
cd deepfake_detection_app
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
pip install -r requirements.txt
🚀 Usage
bash
streamlit run app.py
# Or if using Flask: python app.py
Open your browser at http://localhost:8501 (Streamlit) or http://127.0.0.1:5000 (Flask).

📁 Project Structure
deepfake_detection_app/
├── models/
│   ├── best_val_auc_20260103_201316.h5
│   ├── deploy.prototxt
│   ├── res10_300x300_ssd_iter_140000_fp16.caffemodel
│   └── feature_scalers.pkl
├── templates/
│   └── index.html
├── static/
│   ├── uploads/
│   └── favicon.ico
├── results/
├── app.py
├── model_loader.py
├── video_processor.py
└── requirements.txt
🧠 Model Performance
Hybrid Spatial–Frequency DANN (~8.8M parameters)

Trained on DFDC, FaceForensics++, Celeb‑DF

Cross‑dataset video AUC: 0.9928

Inference: ~12 ms/frame on Tesla T4 GPU

📝 Notes
Decision threshold optimized at 0.39 (max F1 on DFDC‑val)

Robust to JPEG compression and noise

For borderline cases, human inspection is advised

## 📦 Download pre-trained models

- [Download the templates here](https://drive.google.com/drive/folders/1GreU_w4CsETa45GV9TK7jOW_VHEkU_Og?usp=drive_link)

🤝 Contributing
Academic demonstration – issues and pull requests welcome.

📜 License
Educational and research use only. Please cite the original thesis.

🙏 Acknowledgements
Supervisors: Dr Delwende Donald Arthur SAWADOGO, Dr Didier BASSOLE
LAMI Laboratory – UFR‑SEA – Burkina Faso