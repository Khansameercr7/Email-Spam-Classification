# Email Spam Classification Project Report

---

## Personal Information

| Field | Details |
|-------|---------|
| **Full Name** | [Your Name] |
| **Internship Domain** | Machine Learning |
| **Email Address** | [Your Email] |
| **Phone Number** | [Your Phone] |

---

## 1. Project Overview

### 1.1 Introduction
This project implements a machine learning model to classify emails as spam or ham (not spam). The classifier achieves **99.30% accuracy** using Support Vector Machine (SVM) algorithm with TF-IDF feature extraction.

### 1.2 Objectives
- Preprocess email text data (cleaning, stemming, stopword removal)
- Extract important features using TF-IDF vectorization
- Train multiple classification models
- Evaluate model performance with various metrics
- Deploy the model for real-time spam detection

---

## 2. Dataset Description

### 2.1 Data Source
- **File**: `data/emails.csv`
- **Total Records**: 5,728 emails
- **Training Set**: 4,582 samples (80%)
- **Test Set**: 1,146 samples (20%)

### 2.2 Data Structure
| Column | Type | Description |
|--------|------|-------------|
| `text` | String | Email content (subject + body) |
| `spam` | Integer | Label (1 = spam, 0 = ham) |

### 2.3 Class Distribution
- **Ham (Not Spam)**: 872 samples (76.1%)
- **Spam**: 274 samples (23.9%)

---

## 3. Implementation

### 3.1 Preprocessing Pipeline
Located in [src/preprocess.py](src/preprocess.py):

```python
# filepath: src/preprocess.py
def clean_text(text, lowercase=True, remove_punctuation=True, 
               remove_stopwords=True, stemming=True, min_word_length=2):
    """Clean and preprocess email text."""
    
    # Lowercase conversion
    if lowercase:
        text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove punctuation
    if remove_punctuation:
        text = re.sub(r'[^\w\s]', '', text)
    
    # Remove stopwords
    if remove_stopwords:
        text = ' '.join([w for w in text.split() 
                       if w not in STOPWORDS])
    
    # Apply stemming
    if stemming:
        text = ' '.join([stemmer.stem(w) for w in text.split()])
    
    # Filter short words
    if min_word_length:
        text = ' '.join([w for w in text.split() 
                       if len(w) >= min_word_length])
    
    return text
```

### 3.2 Feature Engineering
Located in [src/features.py](src/features.py):

```python
# filepath: src/features.py
def build_vectorizer(method='tfidf', max_features=8000, 
                     ngram_range=(1, 2), sublinear_tf=True):
    """Build TF-IDF or Bag-of-Words vectorizer."""
    
    if method == 'tfidf':
        return TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=sublinear_tf
        )
    else:
        return CountVectorizer(
            max_features=max_features,
            ngram_range=ngram_range
        )
```

### 3.3 Model Training
Located in [src/train.py](src/train.py):

```python
# filepath: src/train.py
def train_all(cfg):
    """Train multiple classifiers and save the best model."""
    
    # Load and preprocess data
    raw = pd.read_csv(cfg["data"]["raw_path"])
    cleaned = preprocess_dataframe(raw, ...)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(...)
    
    # Vectorize
    vec = build_vectorizer(method='tfidf', ...)
    Xtr = fit_transform(vec, X_train)
    Xte = transform(vec, X_test)
    
    # Train models
    models = {
        'MultinomialNB': MultinomialNB,
        'LogisticRegression': LogisticRegression,
        'LinearSVC': LinearSVC,
        'RandomForestClassifier': RandomForestClassifier
    }
    
    # Evaluate and select best
    for name, model in models.items():
        clf = model()
        clf.fit(Xtr, y_train)
        y_pred = clf.predict(Xte)
        # ... calculate metrics
```

### 3.4 Prediction Module
Located in [src/predict.py](src/predict.py):

```python
# filepath: src/predict.py
class SpamPredictor:
    """Wraps the saved vectoriser + classifier for easy inference."""
    
    def __init__(self, config_path="config.yaml"):
        cfg = yaml.safe_load(open(config_path))
        self.vectorizer = joblib.load(cfg["paths"]["vectorizer_path"])
        self.model = joblib.load(cfg["paths"]["model_path"])
    
    def predict(self, text):
        """Predict whether email is spam or ham."""
        cleaned = self._clean(text)
        X = self.vectorizer.transform([cleaned])
        prediction = self.model.predict(X)[0]
        return "SPAM" if prediction == 1 else "HAM"
```

---

## 4. Configuration

### 4.1 config.yaml

```yaml
# filepath: config.yaml
data:
  raw_path: data/emails.csv
  train_path: data/train.csv
  test_path: data/test.csv
  text_column: text
  label_column: spam
  test_size: 0.2
  random_state: 42

preprocessing:
  lowercase: true
  remove_punctuation: true
  remove_stopwords: true
  stemming: true
  min_word_length: 2

features:
  method: tfidf
  max_features: 8000
  ngram_range: [1, 2]
  sublinear_tf: true

models:
  - name: naive_bayes
    type: MultinomialNB
    params:
      alpha: 0.1
  - name: svm
    type: LinearSVC
    params:
      C: 1.0
      max_iter: 2000

training:
  best_model: svm
  scoring_metric: f1
```

---

## 5. Results

### 5.1 Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Naive Bayes | 98.69% | 98.14% | 96.35% | 97.24% |
| Logistic Regression | 98.78% | 98.87% | 95.99% | 97.41% |
| **SVM (Best)** | **99.30%** | **98.54%** | **98.54%** | **98.54%** |
| Random Forest | 98.78% | 96.76% | 98.18% | 97.46% |

### 5.2 Classification Report (Best Model)

```
              precision    recall  f1-score   support

         Ham       0.99      1.00      0.99       872
        Spam       0.99      0.96      0.97       274

    accuracy                           0.99      1146
   macro avg       0.99      0.98      0.98      1146
weighted avg       0.99      0.99      0.99      1146
```

### 5.3 Test Results

```
======================================== 19 passed in 1.55s ========================================
```

---

## 6. Project Screenshots

### 6.1 Training Output
![Training Output](reports/console_training.png)

### 6.2 Confusion Matrix
![Confusion Matrix](reports/confusion_matrix.png)

### 6.3 ROC Curve
![ROC Curve](reports/roc_curve.png)

### 6.4 Precision-Recall Curve
![Precision-Recall Curve](reports/precision_recall_curve.png)

### 6.5 Class Distribution
![Class Distribution](reports/class_distribution.png)

### 6.6 Top Features
![Top Features](reports/top_features.png)

---

## 7. Usage Instructions

### 7.1 Running the Project

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python src/train.py

# Evaluate the model
python src/evaluate.py

# Run tests
pytest tests/ -v
```

### 7.2 Making Predictions

```python
from src.predict import SpamPredictor

# Initialize predictor
p = SpamPredictor()

# Single prediction
result = p.predict("Congratulations! You won a free prize!")
print(result)  # → "SPAM"

# Batch prediction
results = p.predict_batch([
    "Free money!",
    "Hey, are we meeting tomorrow?"
])
print(results)  # → ["SPAM", "HAM"]
```

---

## 8. Project Structure

```
email-spam-classifier/
├── config.yaml          # Configuration file
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
├── data/
│   ├── emails.csv       # Raw dataset
│   ├── train.csv        # Training data
│   └── test.csv         # Test data
├── src/
│   ├── preprocess.py    # Text preprocessing
│   ├── features.py      # Feature extraction
│   ├── train.py         # Model training
│   ├── evaluate.py      # Model evaluation
│   └── predict.py       # Prediction module
├── models/
│   ├── spam_classifier.pkl   # Trained model
│   └── vectorizer.pkl        # TF-IDF vectorizer
├── reports/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── precision_recall_curve.png
│   ├── class_distribution.png
│   └── top_features.png
└── tests/
    ├── test_predict.py
    └── test_preprocess.py
```

---

## 9. Conclusion

This email spam classification project successfully demonstrates:
- Complete ML pipeline from data preprocessing to model deployment
- Multiple model comparison and selection
- **99.30% accuracy** with SVM classifier
- Comprehensive evaluation with confusion matrix, ROC curve, and precision-recall curves
- Unit testing with 19 test cases passing

The model is ready for production deployment and can be easily integrated into email systems for real-time spam detection.

---

## 10. References

- Scikit-learn: Machine Learning in Python
- Pandas: Data Analysis and Manipulation
- NLTK: Natural Language Toolkit
- TF-IDF: Term Frequency-Inverse Document Frequency

---

**Report Generated**: May 2026
**Internship Period**: [Your Period]
**Organization**: Arch Technologies