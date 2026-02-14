# Sports vs. Politics Text Classifier (Hybrid Dataset)

This repository contains the complete implementation and technical report for a binary text classifier designed to distinguish between documents related to Sports and Politics. The system was built using advanced feature engineering (TF-IDF) and a comparison of four machine learning models, culminating in a Stacking Ensemble.

**Author:** D24CSA005
**Project Goal:** Classification Accuracy, Robustness, and Comparative Analysis.

---

## 🚀 Features

*   **Hybrid Dataset:** Merges clean, formal **BBC News** data with noisy, informal **20 Newsgroups** data for superior model generalization. (Total $\approx 5,400$ documents).
*   **Advanced Feature Engineering:** Utilizes **TF-IDF** vectors incorporating **Unigrams and Bigrams** ($N$-gram range (1, 2)).
*   **Model Comparison:** Benchmarks **Naive Bayes, SVM (Linear), Random Forest,** and a **Stacking Ensemble**.
*   **Comprehensive Reporting:** Generates a detailed LaTeX report (as `.tex` source) and saves all necessary visualizations (e.g., Confusion Matrices) as `.png` files.
*   **Interactive Testing:** Allows the user to select which trained model to use for real-time classification tests.

## 🛠️ Implementation Details

The core logic resides in `classifier_ultimate.py`.

*   **Feature Vectorization:** `TfidfVectorizer(max_features=8000, ngram_range=(1, 2))`
*   **Ensemble Method:** `StackingClassifier` using Naive Bayes and SVM as base estimators, with Logistic Regression as the final estimator.
*   **Data Mapping:**
    *   Politics $\rightarrow$ Label **0**
    *   Sports $\rightarrow$ Label **1**

## ⚙️ Setup and Execution

Follow these steps to reproduce the results:

### Step 1: Clone the Repository
```bash
git clone [YOUR_REPOSITORY_URL]
cd your-repo-name# NLU-Assignment
```
### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```
### Step 3: Install Dependencies
Use the provided requirements.txt file to install all necessary libraries:
```bash
pip install -r requirements.txt
```
### Step 4: Run the Classifier Script
Execute the main Python file:
```bash
python classifier_ultimate.py
```
The script will automatically download datasets, train models, print the comparative report to the console, and save all visualizations as PNG files.
### Step 5: Interactive Test
After training, the script enters an interactive mode where you can select the model ID (1-4) and test custom text inputs.
## 📊 Results Summary
The final combined training yielded excellent results:
```bash
Model	            Accuracy	F1-Score	Train Time (s)
Naive Bayes	          97.50%	0.975	  ≈0.003 
Stacking Ensemble	    97.31%	0.973	  ≈73.32 
SVM (Linear)	        95.83%	0.958	  ≈15.35 
Random Forest	        95.45%	0.954	  ≈2.64  
```
The model achieved 97.5% accuracy on the combined, highly varied dataset, proving the effectiveness of feature selection and the robust nature of Naive Bayes for this specific binary text classification task.
## 📄 Technical Report
The full, detailed technical report, including methodology, in-depth analysis, and visualization explanations, is provided in the source file:
```bash
Source: main.tex
Visualizations:
combined_class_distribution.png
combined_feature_importance.png
combined_confusion_matrices.png
```
