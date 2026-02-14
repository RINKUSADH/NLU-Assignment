import os
import urllib.request
import zipfile
import pandas as pd
import numpy as np
import re
import sys
import time
import matplotlib.pyplot as plt
import seaborn as sns

# ML Libraries
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_recall_fscore_support

# Setup plotting style
plt.style.use('ggplot')

# ==========================================
# 1. DATA LOADING FUNCTIONS
# ==========================================
def load_bbc_data():
    """Downloads and loads the BBC News dataset."""
    print("\n[1/3] Loading BBC Dataset (Professional News)...")
    DATA_URL = "http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip"
    DATA_DIR = "bbc_data"
    ZIP_FILE = "bbc-fulltext.zip"

    # Download if missing
    if not os.path.exists(DATA_DIR):
        print("   Downloading BBC data...")
        try:
            urllib.request.urlretrieve(DATA_URL, ZIP_FILE)
            with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
                zip_ref.extractall(DATA_DIR)
            if os.path.exists(ZIP_FILE): os.remove(ZIP_FILE)
        except Exception as e:
            print(f"   Error downloading BBC: {e}")
            return pd.DataFrame()

    # Read Files
    texts = []
    labels = []
    base_path = os.path.join(DATA_DIR, 'bbc')
    
    # Handle extracting structure variations
    if not os.path.exists(base_path): base_path = DATA_DIR 

    # 0 = Politics, 1 = Sport
    categories = {'politics': 0, 'sport': 1}
    
    count = 0
    for category, label in categories.items():
        path = os.path.join(base_path, category)
        if os.path.exists(path):
            for filename in os.listdir(path):
                if filename.endswith(".txt"):
                    try:
                        with open(os.path.join(path, filename), 'r', encoding='latin-1') as f:
                            texts.append(f.read())
                            labels.append(label)
                            count += 1
                    except: pass
    
    df = pd.DataFrame({'text': texts, 'label': labels})
    df['source'] = 'BBC (UK)'
    print(f"   Loaded {count} documents from BBC.")
    return df

def load_20news_data():
    """Downloads and loads the 20 Newsgroups dataset."""
    print("\n[2/3] Loading 20 Newsgroups Dataset (Internet Forums)...")
    categories = [
        'rec.sport.baseball', 'rec.sport.hockey',  # SPORTS
        'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc' # POLITICS
    ]
    
    try:
        dataset = fetch_20newsgroups(subset='all', categories=categories, 
                                     remove=('headers', 'footers', 'quotes'))
        
        df = pd.DataFrame({'text': dataset.data, 'target': dataset.target})
        
        # Mapping: Indices 0,1 are Sports. Indices 2,3,4 are Politics.
        # We map: Sport -> 1, Politics -> 0
        df['label'] = df['target'].apply(lambda x: 1 if x < 2 else 0)
        
        # Filter short texts (noise removal)
        df = df[df['text'].str.len() > 20]
        df['source'] = '20News (US)'
        
        print(f"   Loaded {len(df)} documents from 20 Newsgroups.")
        return df[['text', 'label', 'source']]
    except Exception as e:
        print(f"   Error loading 20News: {e}")
        return pd.DataFrame()

# ==========================================
# 2. COMBINE & VISUALIZE DATA
# ==========================================
def prepare_combined_data():
    df_bbc = load_bbc_data()
    df_news = load_20news_data()
    
    if df_bbc.empty and df_news.empty:
        print("Error: No data loaded.")
        sys.exit()

    # Concatenate
    print("\n[3/3] Merging and Shuffling Datasets...")
    df_final = pd.concat([df_bbc, df_news], ignore_index=True)
    
    # Shuffle
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Add readable label
    df_final['Category'] = df_final['label'].apply(lambda x: 'Sport' if x == 1 else 'Politics')
    
    print(f"TOTAL DOCUMENTS: {len(df_final)}")
    print(df_final['Category'].value_counts())
    
    # --- VIZ 1: DISTRIBUTION ---
    plt.figure(figsize=(10, 6))
    sns.countplot(x='source', hue='Category', data=df_final, palette='viridis')
    plt.title('Combined Dataset Distribution')
    print(">> Saving plot to 'combined_class_distribution.png'")
    plt.savefig('combined_class_distribution.png')
    plt.close()
    
    return df_final

# ==========================================
# 3. FEATURES & PREPROCESSING
# ==========================================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text) 
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_features(df):
    print("\n--- Feature Extraction (TF-IDF) ---")
    df['clean_text'] = df['text'].apply(clean_text)
    
    # Max features 8000 for larger vocabulary
    tfidf = TfidfVectorizer(stop_words='english', max_features=8000, ngram_range=(1,2))
    X = tfidf.fit_transform(df['clean_text'])
    y = df['label']
    
    # --- VIZ 2: FEATURE IMPORTANCE ---
    sum_words = X.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in tfidf.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)[:20]
    top_words_df = pd.DataFrame(words_freq, columns=['Word', 'Weight'])
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Weight', y='Word', data=top_words_df, hue='Word', palette='magma', legend=False)
    plt.title('Top 20 Features in Combined Dataset')
    print(">> Saving plot to 'combined_feature_importance.png'")
    plt.savefig('combined_feature_importance.png')
    plt.close()
    
    return X, y, tfidf

# ==========================================
# 4. TRAINING & EVALUATION
# ==========================================
def train_and_report(X, y):
    print("\n" + "="*50)
    print("TRAINING MODELS (This may take a minute)...")
    print("="*50)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Models
    nb = MultinomialNB()
    svm = SVC(kernel='linear', probability=True, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Stacking
    estimators = [('nb', nb), ('svm', svm)]
    stacking = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

    models = {
        "Naive Bayes": nb,
        "SVM (Linear)": svm,
        "Random Forest": rf,
        "Stacking Ensemble": stacking
    }
    
    log = []
    confusion_matrices = {}
    
    for name, model in models.items():
        print(f"Training {name}...", end=" ")
        
        # Train Time
        t0 = time.time()
        model.fit(X_train, y_train)
        t_train = time.time() - t0
        
        # Test Time
        t1 = time.time()
        y_pred = model.predict(X_test)
        t_test = time.time() - t1
        
        print(f"Done ({t_train:.2f}s)")
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        cm = confusion_matrix(y_test, y_pred)
        
        confusion_matrices[name] = cm
        log.append({
            "Model": name, "Accuracy": acc, "Precision": prec, 
            "Recall": rec, "F1-Score": f1, 
            "Train Time": t_train, "Test Time": t_test,
            "Object": model
        })

    # --- REPORT ---
    results_df = pd.DataFrame(log).set_index("Model")
    print("\n" + "="*80)
    print("FINAL COMPARISON REPORT (Combined Data)")
    print("="*80)
    print(results_df[['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Train Time']])
    print("="*80)
    
    # --- VIZ 3: CONFUSION MATRICES ---
    print(">> Saving plot to 'combined_confusion_matrices.png'")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    for ax, (name, cm) in zip(axes, confusion_matrices.items()):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Politics', 'Sport'], yticklabels=['Politics', 'Sport'])
        ax.set_title(name)
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('combined_confusion_matrices.png')
    plt.close()
    
    return results_df

# ==========================================
# 5. INTERACTIVE MENU
# ==========================================
def interactive_menu(results_df, tfidf):
    print("\n" + "="*50)
    print("INTERACTIVE MODE - SELECT A MODEL")
    print("="*50)
    
    model_map = {str(i+1): name for i, name in enumerate(results_df.index)}
    
    while True:
        print("\nAvailable Models:")
        for k, v in model_map.items():
            print(f"[{k}] {v}")
        print("[Q] Quit")
        
        choice = input("Select Model ID: ").strip().upper()
        if choice == 'Q': 
            print("Exiting...")
            break
        
        if choice not in model_map:
            print("Invalid choice.")
            continue
            
        model_name = model_map[choice]
        model = results_df.loc[model_name, "Object"]
        
        print(f"\n>> Loaded: {model_name}")
        print("Type a headline to classify (or 'back' to choose another model).")
        
        while True:
            text = input(f"({model_name}) Enter text: ")
            
            if text.lower() == 'back': break
            if text.lower() in ['quit', 'exit', 'q']: 
                print("Goodbye!")
                sys.exit()
            
            if len(text) < 2: continue
            
            # Predict
            vec = tfidf.transform([clean_text(text)])
            pred = model.predict(vec)[0]
            
            # Try to get probability if supported
            try:
                prob = model.predict_proba(vec)[0].max()
                conf_str = f"(Confidence: {prob*100:.1f}%)"
            except:
                conf_str = ""

            label = "SPORT" if pred == 1 else "POLITICS"
            
            print(f"Prediction: [{label}] {conf_str}")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Load Combined Data
    df = prepare_combined_data()
    
    # 2. Extract Features
    X, y, tfidf = extract_features(df)
    
    # 3. Train & Report
    results = train_and_report(X, y)
    
    # 4. Interactive Menu
    interactive_menu(results, tfidf)
