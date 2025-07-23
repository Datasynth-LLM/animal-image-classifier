# 🐾 Animal Image Classifier

A deep learning-based image classification project that identifies animals from images using Convolutional Neural Networks (CNNs) with Transfer Learning and PyTorch. This project is built with modular architecture and a friendly Streamlit UI for end-user interaction.

---

## 📌 Project Objective

> Build a system that classifies an image into one of **15 animal categories** using a trained model on a provided dataset of 224x224 RGB images.

---

## 🧠 Dataset Details

- Each folder in the dataset corresponds to an animal class: **Bear**, **Bird**, **Cat**, **Cow**, **Deer**, **Dog**, **Dolphin**, **Elephant**, **Giraffe**, **Horse**, **Kangaroo**, **Lion**, **Panda**, **Tiger**, **Zebra**.
- Image resolution: `224 x 224 x 3`
- Data is structured in:
  ```
  data/processed/
    ├── train/
    ├── val/
    └── test/
  ```

---

## ✅ Features Implemented

| Feature                   | Status | Description                                                         |
| ------------------------- | ------ | ------------------------------------------------------------------- |
| 🔍 EDA Notebook           | ✅      | Analysis & visualization using `eda.ipynb`                          |
| 🧠 CNN Model              | ✅      | Custom-built CNN model in `src/models/cnn.py`                       |
| 🧪 Training Logic         | ✅      | Train on GPU/CPU with early model saving                            |
| 📂 Modular Code Structure | ✅      | Code is organized by `src/`, `scripts/`, `models/` etc.             |
| 📊 Evaluation             | ✅      | Generates classification report and confusion matrix                |
| 📉 Training Metrics       | ✅      | Saves loss and accuracy plots                                       |
| 🎨 Streamlit UI           | ✅      | User-friendly animal image classification web app                   |
| 🎆 Dark Theme UI          | ✅      | Custom purple-themed Streamlit interface                            |
| 🗂️ Batch Script          | ✅      | `run.bat` file launches Streamlit first, then training, then report |
| 🧪 Backup System          | ✅      | Backs up previous model before retraining                           |
| 📈 Top-3 Predictions      | ✅      | UI shows confidence scores for top 3 classes                        |
| 📁 Reporting Automation   | ✅      | Saves classification report and confusion matrix automatically      |

---

## 📁 Project Structure

```
animal_classifier_unified_mentor/
├── app/                    # Streamlit app
├── data/                   # Raw & processed image data
├── models/                 # Saved CNN models
├── notebooks/              # EDA, experiments
├── reports/                # Confusion matrix, plots, metrics
├── scripts/                # Training/report generation logic
├── src/                    # Model + training source
│   ├── models/
│   └── training/
├── evaluate.py             # Evaluation script
├── requirements.txt        # Dependencies
├── .streamlit/config.toml  # Custom dark purple theme
├── run.bat                 # Main automation batch script
└── README.md               # This file
```

---

## ⚙️ How to Run

### 📦 1. Setup Environment

```bash
python -m venv venv
venv\Scripts\activate     # For Windows
pip install -r requirements.txt
```

> 💡 If any install fails, run `pip install <package>` manually.

---

### ▶️ 2. Run the Full Pipeline (Auto Mode)

Use the `run.bat` file to:

- Launch Streamlit app immediately
- Then train the model in the background
- Then generate reports

```bash
run.bat
```

---

### 🖥️ 3. Manual Streamlit Launch

```bash
cd app
streamlit run app.py
```

Visit: [http://localhost:8501](http://localhost:8501)

---

### 🧪 4. Run Scripts Manually (Optional)

```bash
# Train model
python src/training/train.py

# Generate reports
python scripts/generate_report.py

# Evaluate and view confusion matrix
python evaluate.py
```

---

## 🛠️ Tech Stack

- **Python 3.11**
- **PyTorch** – Model definition & training
- **TorchVision** – Data transforms, image handling
- **Streamlit** – Web UI
- **Matplotlib / Seaborn** – Evaluation plots
- **scikit-learn** – Metrics and confusion matrix

---

## 🔎 Implementation Narrative

1. **Data Loading**: Images were organized into `train`, `val`, and `test` folders and loaded using `ImageFolder`.
2. **Modeling**: A custom CNN architecture (`CustomCNN`) was designed and trained using cross-entropy loss and Adam optimizer.
3. **Training**: The script saved the best model (`best_model.pth`) based on validation accuracy. Previous models were backed up automatically.
4. **Evaluation**: After training, `evaluate.py` generated the confusion matrix and classification report using unseen test data.
5. **Report Generation**: `generate_report.py` automates saving plots of accuracy/loss across epochs and summary reports.
6. **Web App**: A Streamlit interface lets users upload animal images and get top-3 predictions with confidence scores and emoji indicators.
7. **Theme Customization**: The app uses a sleek purple dark mode defined in `.streamlit/config.toml`.
8. **Batch Automation**: `run.bat` handles full workflow including app start, background training, and report generation.

---

## 📌 Notes

- If `best_model.pth` exists, training backs it up first to avoid overwriting.
- Make sure images in all folders are 224x224 or use resizing.
- The model handles rotated or slightly skewed images fairly well thanks to data augmentation.

---

## 🙌 Credits

Built as part of a Deep Learning Internship task — mentored by **Unified Mentors**.\
All implementation, model design, and automation workflows were created by the contributor.

---

## 🚀 Ready to Demo?

Just run:

```bash
run.bat
```

Then open the link in your browser and start classifying! 🐻 🐯 🦓

