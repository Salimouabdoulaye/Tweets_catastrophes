import re
import contractions

# --- Installées
# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Traitement des données
import pandas as pd
import numpy as np

# NLP
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingClassifier


# Sauvegarde des modèles
import joblib

# Système et utilitaires
import os
from datetime import datetime
from pathlib import Path

# API et web
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# Implicite 
import streamlit
import requests
import uvicorn

# Téléchargement des ressources NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Configuration des stop words
stop_words = set(stopwords.words('english'))

# Définir le répertoire de base du projet
BASE_DIR = Path.cwd()
print(f"Dossier de base : {BASE_DIR}")

# Dossier contenant les données
DATA_DIR = BASE_DIR / 'data'
print(f"Dossier de données : {DATA_DIR}")

# Dossier des résultats horodaté
RESULTS_DIR = BASE_DIR / 'results' / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
print(f"Dossier de résultats : {RESULTS_DIR}")

# Dossier pour sauvegarder les figures
FIGURES_DIR = RESULTS_DIR / 'figures'
print(f"Dossier de figures : {FIGURES_DIR}")

# Création des dossiers si besoin
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def preprocess_text(text, language="english", stopwords_list=None, normalizer="stem", 
                    remove_numbers=True, expand_contractions=True, keep_emojis=False):
    """
    Nettoie et normalise le texte .
    
    Args:
        text (str): Texte brut à prétraiter.
        language (str): Langue pour stemming/stopwords.
        stopwords_list (set): Liste de mots à supprimer.
        normalizer (str): 'stem', 'lemma' ou None pour normalisation.
        remove_numbers (bool): Supprimer les nombres.
        expand_contractions (bool): Développer les contractions (ex. "don't" -> "do not").
        keep_emojis (bool): Conserver les emojis.
    
    Returns:
        str: Texte nettoyé et normalisé.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    
    text = text.lower()
    if expand_contractions:
        text = contractions.fix(text)
    text = re.sub(r"<.*?>", "", text)  # HTML
    text = re.sub(r"(http|www)\S*", "", text)  # URL
    text = re.sub(r"\S*@\S*\s*", "", text)  # Email
    if not keep_emojis:
        text = re.sub(r"[^\w\s]", " ", text)  # Supprime ponctuation et emojis
    else:
        text = re.sub(r"[^\w\s\p{Emoji}]", " ", text, flags=re.UNICODE)  # Conserve emojis
    if remove_numbers:
        text = re.sub(r"\d+", "", text).strip()  # Numbers + trim
    
    tokens = word_tokenize(text)
    if stopwords_list:
        tokens = [word for word in tokens if word not in stopwords_list]
    
    if normalizer == "stem":
        stemmer = SnowballStemmer(language)
        tokens = [stemmer.stem(w) for w in tokens]
    elif normalizer == "lemma":
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(w) for w in tokens]
    
    return " ".join(tokens)