## Email Spam Classifier

Welcome! This project was created by Sameer Khan as part of a hands-on machine learning journey. The goal: build a real-world email spam detector from scratch, using a public dataset and open-source tools.

---

## 📖 Project Story

I wanted to understand how spam filters work under the hood, so I set out to:
- Clean and preprocess messy email text
- Extract meaningful features (words, n-grams)
- Train and compare several ML models
- Build a simple web app for anyone to try it out

This repo is the result—a practical, end-to-end solution, not just a demo.

---

## 📁 Project Structure

```
email-spam-classifier/
├── app.py                  # Streamlit web app (try it yourself!)
├── config.yaml             # All settings in one place
├── requirements.txt        # Python dependencies
├── data/                   # Data files (from Kaggle)
│   ├── emails.csv          # Raw dataset
│   ├── train.csv           # Training set
│   └── test.csv            # Test set
├── models/                 # Saved models
│   ├── spam_classifier.pkl
│   └── vectorizer.pkl
├── notebooks/              # Jupyter notebooks (step-by-step)
│   ├── 01_eda.ipynb        # Data exploration
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_model_evaluation.ipynb
├── reports/                # Evaluation reports
├── src/                    # Source code
│   ├── preprocess.py       # Text cleaning
│   ├── features.py         # Feature extraction
│   ├── train.py            # Model training
│   ├── evaluate.py         # Model evaluation
│   └── predict.py          # Prediction API
└── tests/                  # Unit tests
    ├── test_preprocess.py
    └── test_predict.py
```

---

## 🚀 How to Use

1. **Install dependencies**
   ```powershell
   pip install -r requirements.txt
   ```
2. **Train the models**
   ```powershell
   python src/train.py
   ```
3. **Run the web app**
   ```powershell
   streamlit run app.py
   ```
4. **Run tests**
   ```powershell
   pytest tests/ -v
   ```

---

## ⚙️ Configuration

All settings are in `config.yaml`—change models, features, or preprocessing in one place.

---

## 📊 Models Compared

- Naive Bayes
- Logistic Regression
- SVM
- Random Forest
- Voting Ensemble (combines all)

---

## 🖥️ Try It Out

**Web App:** Paste any email text and see if it’s spam or not.

**Python API:**
```python
from src.predict import SpamPredictor
predictor = SpamPredictor()
print(predictor.predict("Win a free iPhone now!"))
```

**CLI:**
```bash
python src/predict.py "Congratulations! You have won a prize!"
```

---

## 📈 Results & Insights

- 5-fold cross-validation for robust scores
- Hyperparameter tuning (GridSearchCV)
- Confusion matrix and ROC curve visualizations

---

## 📝 Limitations & Future Work

- Only English emails are supported
- No deep learning (for simplicity)
- Could add batch upload, email parsing, or explainable AI

---

## 👤 About the Author

**Sameer Khan**  
Aspiring data scientist, passionate about real-world ML.  
Email: khansameercr7@gmail.com

---

## 📷 Screenshots

_Add screenshots of the web app here for a personal touch!_

---

## 🛠️ Requirements

- Python 3.8+
- scikit-learn
- pandas
- numpy
- streamlit

See `requirements.txt` for the full list.
=======
# Email-Spam-Classification
This project implements a machine learning model to classify emails as spam or ham (not spam). The classifier achieves **99.30% accuracy** using Support Vector Machine (SVM) algorithm with TF-IDF feature extraction.
>>>>>>> 1d770a57c7d608ad4d08defc25c6c9a6860a5744
