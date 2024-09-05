# 📰 NewsClassify-NGram: News Article Classification using Bag of N-grams

## 🌟 Project Overview

NewsClassify-NGram is a machine learning project that classifies news articles into various categories using the bag-of-n-grams approach. This Jupyter notebook demonstrates the use of natural language processing techniques and machine learning algorithms to automatically categorize news articles based on their content.

## 🚀 Features

- 📊 Utilizes scikit-learn's CountVectorizer for creating bag-of-n-grams representations
- 🧹 Implements text preprocessing using spaCy for improved classification accuracy
- ⚖️ Handles class imbalance in the dataset
- 🔬 Compares performance of different n-gram ranges (1-gram, 2-gram, 3-gram)
- 🧠 Uses Multinomial Naive Bayes classifier for text classification
- 📈 Provides detailed classification reports for model evaluation

## 📋 Prerequisites

- Python 3.x
- Jupyter Notebook
- pandas
- scikit-learn
- spaCy
- numpy

You'll also need to download the English language model for spaCy:

```python
!python -m spacy download en_core_web_sm
```

## 📊 Dataset

The project uses a JSON file named `news_dataset.json` containing news articles. Here's a breakdown of the dataset:

```python
import pandas as pd

df = pd.read_json('news_dataset.json')
print(df.shape)
```

Output:
```
(12695, 2)
```

The dataset contains 12,695 news articles with two columns:
- `text`: The content of the news article
- `category`: The category of the news article

Let's look at the first few rows:

```python
df.head()
```

Output:
```
                                                text category
0  Watching Schrödinger's Cat Die University of C...  SCIENCE
1  WATCH: Freaky Vortex Opens Up In Flooded Lake    SCIENCE
2  Entrepreneurs Today Don't Need a Big Budget to...  BUSINESS
3  These Roads Could Recharge Your Electric Car A...  BUSINESS
4  Civilian 'Guard' Fires Gun While 'Protecting' ...    CRIME
```

The distribution of categories:

```python
df.category.value_counts()
```

Output:
```
BUSINESS    4254
SPORTS      4167
CRIME       2893
SCIENCE     1381
Name: count, dtype: int64
```

## 🧹 Data Preprocessing

The project uses spaCy for text preprocessing:

```python
import spacy

nlp = spacy.load("en_core_web_sm")

def preprocess(text):
    doc = nlp(text)
    filtered_tokens = []
    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        filtered_tokens.append(token.lemma_)
    return " ".join(filtered_tokens)
```

This function removes stop words and punctuation, and lemmatizes the text.

## ⚖️ Handling Class Imbalance

To address the class imbalance, we undersample the majority classes:

```python
min_samples = 1381  # number of samples in the minority class (SCIENCE)

df_balanced = pd.concat([
    df[df.category=="BUSINESS"].sample(min_samples, random_state=2022),
    df[df.category=="SPORTS"].sample(min_samples, random_state=2022),
    df[df.category=="CRIME"].sample(min_samples, random_state=2022),
    df[df.category=="SCIENCE"]
])
```

## 🧠 Model Training and Evaluation

The project uses scikit-learn's Pipeline to combine vectorization and classification:

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

clf = Pipeline([
    ('vectorizer_bow', CountVectorizer(ngram_range=(1, 1))),
    ('Multi NB', MultinomialNB())
])

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
```

## 📊 Results

Here are the classification reports for different n-gram ranges:

1. Bag of Words (1-gram):
```
              precision    recall  f1-score   support

           0       0.75      0.87      0.81       276
           1       0.93      0.80      0.86       277
           2       0.83      0.90      0.86       276
           3       0.90      0.80      0.85       276

    accuracy                           0.84      1105
   macro avg       0.85      0.84      0.84      1105
weighted avg       0.85      0.84      0.84      1105
```

2. Bag of 1-2 grams:
```
              precision    recall  f1-score   support

           0       0.69      0.90      0.78       276
           1       0.95      0.74      0.83       277
           2       0.82      0.88      0.85       276
           3       0.92      0.78      0.84       276

    accuracy                           0.82      1105
   macro avg       0.85      0.82      0.83      1105
weighted avg       0.85      0.82      0.83      1105
```

3. Bag of 1-3 grams:
```
              precision    recall  f1-score   support

           0       0.67      0.91      0.77       276
           1       0.96      0.73      0.83       277
           2       0.83      0.87      0.85       276
           3       0.93      0.76      0.83       276

    accuracy                           0.82      1105
   macro avg       0.84      0.82      0.82      1105
weighted avg       0.84      0.82      0.82      1105
```

## 🏆 Best Model Results: Bag of 1-2 grams with Text Preprocessing

### Model Configuration

This model uses a combination of text preprocessing and bag of 1-2 grams, which proved to be the most effective approach in our news classification task.

### Preprocessing Step

```python
import spacy

nlp = spacy.load("en_core_web_sm")

def preprocess(text):
    doc = nlp(text)
    filtered_tokens = []
    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        filtered_tokens.append(token.lemma_)
    return " ".join(filtered_tokens)

df_balanced['preprocessed_txt'] = df_balanced['text'].apply(preprocess)
```

### Model Pipeline

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

clf = Pipeline([
    ('vectorizer_bow', CountVectorizer(ngram_range=(1, 2))),
    ('Multi NB', MultinomialNB())
])

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
```

## Classification Report

```
              precision    recall  f1-score   support

           0       0.80      0.88      0.84       276
           1       0.92      0.82      0.87       277
           2       0.83      0.92      0.87       276
           3       0.90      0.81      0.85       276

    accuracy                           0.86      1105
   macro avg       0.86      0.86      0.86      1105
weighted avg       0.86      0.86      0.86      1105
```

## Results Breakdown

- **Overall Accuracy**: 0.86 (86%)

- **Per-Class Performance**:
  - Class 0 (BUSINESS):
    - Precision: 0.80
    - Recall: 0.88
    - F1-score: 0.84
  - Class 1 (SPORTS):
    - Precision: 0.92
    - Recall: 0.82
    - F1-score: 0.87
  - Class 2 (CRIME):
    - Precision: 0.83
    - Recall: 0.92
    - F1-score: 0.87
  - Class 3 (SCIENCE):
    - Precision: 0.90
    - Recall: 0.81
    - F1-score: 0.85

## Analysis

1. **Balanced Performance**: The model shows consistent performance across all classes, with F1-scores ranging from 0.84 to 0.87. This indicates that the preprocessing and balanced dataset have helped in achieving uniform classification across categories.

2. **High Precision for SPORTS**: The model is particularly good at identifying SPORTS articles correctly, with a precision of 0.92. This means when it predicts an article is about sports, it's right 92% of the time.

3. **High Recall for CRIME**: The model catches 92% of all CRIME articles in the dataset, showing high sensitivity for this category.

4. **Room for Improvement**: While the overall performance is good, there's still room for improvement, especially in the recall for SCIENCE articles and precision for BUSINESS articles.

5. **Effectiveness of Preprocessing**: The improvement in performance compared to models without preprocessing demonstrates the value of the text cleaning and lemmatization steps.

This model's performance suggests that the combination of text preprocessing and using 1-2 grams effectively captures the distinctive features of each news category, leading to accurate classification.

## 🚀 Future Improvements

- Experiment with other classification algorithms (e.g., SVM, Random Forest)
- Implement cross-validation for more robust evaluation
- Try more advanced text representation techniques (e.g., word embeddings, transformers)
- Develop a web interface for easy article classification

