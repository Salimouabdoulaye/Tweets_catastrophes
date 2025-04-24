# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, ConfusionMatrixDisplay
from sklearn.metrics import classification_report, recall_score, f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, StratifiedKFold
from transformers import AutoTokenizer, AutoModel
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import torch
from tqdm import tqdm
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
import pickle
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
import torch


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

import pandas as pd
import numpy as np
import json
import pickle


class BERTEmbedder(BaseEstimator, TransformerMixin):
    """
    Classe pour transformer des textes en embeddings BERT
    Compatible avec les pipelines sklearn
    """
    def __init__(self, model_name='bert-base-multilingual-cased', max_length=128, embedding_strategy="cls"):
        """
        Initialise la classe BERTEmbedder
        
        Args:
            model_name (str): Nom du modèle à utiliser
            max_length (int): Longueur maximale des séquences après tokenization
            embedding_strategy (str): Stratégie d'embedding ("cls" ou "mean")
        """
        self.model_name = model_name
        self.max_length = max_length
        self.embedding_strategy = embedding_strategy
        self.tokenizer = None
        self.model = None
        
        valid_emb_strategy = ("cls", "mean")
        if self.embedding_strategy not in valid_emb_strategy:
            raise ValueError(f"embedding_strategy doit être l'un des suivants: {valid_emb_strategy}")

    def fit(self, X, y=None):
        """ Charge le modèle et le tokenizer
        Args:
            X: Les textes d'entrée
            y: Les labels
            
        Returns:
            self: Retourne l'instance pour le chaînage
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.eval()
        return self

    def transform(self, X):
        """
        Transforme les textes en embeddings
        
        Args:
            X: Liste de textes à transformer
            
        Returns:
            np.array: Matrice des embeddings
        """
        # Convertir en liste si nécessaire
        if isinstance(X, pd.Series) or isinstance(X, np.ndarray):
            X = X.tolist()
        elif not isinstance(X, list):
            raise ValueError("Les données d'entrée doivent être une liste ou convertible en liste de chaînes de caractères.")
        
        # Nettoyage et conversion en string
        X_cleaned = [str(text) if not pd.isna(text) else "" for text in X]
        
        embeddings = []
        
        for text in tqdm(X_cleaned, desc=f"Embeddings {self.model_name}"):
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, 
                                    padding=True, max_length=self.max_length)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            last_hidden_states = outputs.last_hidden_state
            
            if self.embedding_strategy == 'cls':
                embedding = last_hidden_states[:, 0, :].squeeze().numpy()
            else:  # mean
                embedding = last_hidden_states.mean(dim=1).squeeze().numpy()
            
            embeddings.append(embedding)
        
        return np.vstack(embeddings)

def test_multiple_classifiers(model_name, twenty_train, twenty_test, embedding_strategy="cls"):
    """
    Crée des pipelines sklearn avec BERTEmbedder et différents classifieurs,
    les entraîne et les évalue
    
    Args:
        model_name (str): Nom du modèle BERT à utiliser
        twenty_train: Données d'entraînement
        twenty_test: Données de test
        embedding_strategy (str): Stratégie d'embedding ("cls" ou "mean")
        
    Returns:
        tuple: (meilleur_classifieur, scores)
    """
    print(f"\n=== Test du modèle {model_name} (embedding par: {embedding_strategy}) ===")
    
    # Assurer que processed_text est une liste de strings
    X_train = twenty_train.processed_text.tolist() if hasattr(twenty_train, 'processed_text') else twenty_train.tolist()
    X_test = twenty_test.processed_text.tolist() if hasattr(twenty_test, 'processed_text') else twenty_test.tolist()
    
    y_train = twenty_train.target if hasattr(twenty_train, 'target') else twenty_train
    y_test = twenty_test.target if hasattr(twenty_test, 'target') else twenty_test
    
    # Créer l'embedder BERT
    embedder = BERTEmbedder(model_name=model_name, max_length=128, embedding_strategy=embedding_strategy)
    
    # Transformer les données une seule fois pour éviter de répéter cette opération coûteuse
    print("Création des embeddings (peut prendre du temps)...")
    X_train_embedded = embedder.fit_transform(X_train)
    X_test_embedded = embedder.transform(X_test)
    
    # Définir les classifieurs à tester
    classifiers = {
        'LogisticRegression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        'SVM': SVC(class_weight='balanced', random_state=42, probability=True),
        'XGBoost': xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
        'LightGBM': lgb.LGBMClassifier(random_state=42, class_weight='balanced')
    }
    
    results = {}
    
    # Tester chaque classifieur
    for name, clf in classifiers.items():
        print(f"\nEntraînement du modèle {name}...")
        clf.fit(X_train_embedded, y_train)
        y_pred = clf.predict(X_test_embedded)
        
        # Évaluer les performances avec focus sur la classe 1
        accuracy = (y_pred == y_test).mean()
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        recall_class1 = recall_score(y_test, y_pred, average=None)[1] if 1 in np.unique(y_test) else 0
        f1_class1 = f1_score(y_test, y_pred, average=None)[1] if 1 in np.unique(y_test) else 0
        
        results[name] = {
            'accuracy': accuracy,
            'recall': recall,
            'f1': f1,
            'recall_class1': recall_class1,
            'f1_class1': f1_class1,
            'clf': clf
        }
        
        print(f"Résultats pour {name}:")
        print(f"Exactitude: {accuracy:.4f}")
        print(f"Recall (général): {recall:.4f}")
        print(f"F1-score (général): {f1:.4f}")
        print(f"Recall (classe 1): {recall_class1:.4f}")
        print(f"F1-score (classe 1): {f1_class1:.4f}")
        print(classification_report(y_test, y_pred))
    
    # Déterminer le meilleur classifieur en fonction du F1-score et recall pour classe 1
    best_score = 0
    best_classifier = None
    
    for name, scores in results.items():
        # Score combiné donnant un poids égal au recall et F1 pour la classe 1
        score = (scores['f1_class1'])
        if score > best_score:
            best_score = score
            best_classifier = name
    
    print(f"\nMeilleur classifieur: {best_classifier} avec un score de {best_score:.4f}")
    
    # Créer une pipeline complète avec le meilleur classifieur
    best_pipeline = Pipeline([
        ('embedder', embedder),
        ('classifier', results[best_classifier]['clf'])
    ])
    
    return best_classifier, results, best_pipeline, embedder

def optimize_best_classifier(best_classifier, X_train_embedded, y_train, X_test_embedded, y_test):
    """
    Optimise les hyperparamètres du meilleur classifieur
    
    Args:
        best_classifier (str): Nom du meilleur classifieur
        X_train_embedded: Données d'entraînement embedées
        y_train: Labels d'entraînement
        X_test_embedded: Données de test embedées
        y_test: Labels de test
        
    Returns:
        model: Modèle optimisé
    """
    print(f"\n=== Optimisation du classifieur {best_classifier} ===")
    
    # Définir les grilles de paramètres pour chaque classifieur
    param_grids = {
        'LogisticRegression': {
            'C': [0.01, 0.1, 1, 10, 100],
            'solver': ['liblinear', 'lbfgs'],
            'penalty': ['l1', 'l2']
        },
        'SVM': {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto', 0.1, 0.01],
            'kernel': ['rbf', 'linear']
        },
        'XGBoost': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3]
        },
        'LightGBM': {
            'n_estimators': [50, 100, 200],
            'num_leaves': [31, 50, 100],
            'learning_rate': [0.01, 0.1, 0.3]
        }
    }
    
    # Sélectionner le classifieur et la grille de paramètres appropriés
    if best_classifier == 'LogisticRegression':
        clf = LogisticRegression(class_weight='balanced', random_state=42, max_iter=2000)
    elif best_classifier == 'SVM':
        clf = SVC(class_weight='balanced', random_state=42, probability=True)
    elif best_classifier == 'XGBoost':
        clf = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    elif best_classifier == 'LightGBM':
        clf = lgb.LGBMClassifier(random_state=42, class_weight='balanced')
    else:
        raise ValueError(f"Classifieur {best_classifier} non reconnu")
    
    param_grid = param_grids[best_classifier]
    
    # Optimisation par validation croisée
    grid_search = GridSearchCV(
        clf, param_grid,
        cv=3,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=2
    )
    
    grid_search.fit(X_train_embedded, y_train)
    
    # Récupérer le meilleur modèle
    best_model = grid_search.best_estimator_
    print(f"Meilleurs paramètres: {grid_search.best_params_}")
    
    # Évaluer le modèle optimisé
    y_pred = best_model.predict(X_test_embedded)
    
    print("\nRésultats après optimisation:")
    print(f"Exactitude: {(y_pred == y_test).mean():.4f}")
    print(f"Recall (classe 1): {recall_score(y_test, y_pred, average=None)[1] if 1 in np.unique(y_test) else 0:.4f}")
    print(f"F1-score (classe 1): {f1_score(y_test, y_pred, average=None)[1] if 1 in np.unique(y_test) else 0:.4f}")
    print(classification_report(y_test, y_pred))
    
    return best_model

def run_bert_experiments(twenty_train, twenty_test):
    """
    Exécute des expériences avec différents modèles BERT, stratégies d'embedding
    et différents classifieurs
    
    Args:
        twenty_train: Données d'entraînement
        twenty_test: Données de test
        
    Returns:
        tuple: (meilleur_modèle, meilleur_embedder)
    """
    models = [
        'bert-base-multilingual-cased',  # BERT multilingue (pour plusieurs langues)
        'distilbert-base-uncased',       # DistilBERT (version plus légère)
    ]
    
    strategies = ['mean', 'cls']  # Stratégies d'embedding: 'mean' ou 'cls'
    
    results = {}
    best_pipelines = {}
    
    for model in models:
        for strategy in strategies:
            model_key = f"{model}_{strategy}"
            best_clf, clf_results, pipeline, embedder = test_multiple_classifiers(
                model, twenty_train, twenty_test, embedding_strategy=strategy
            )
            
            # Stocker les résultats
            results[model_key] = {
                'best_classifier': best_clf,
                'classifier_results': clf_results
            }
            
            best_pipelines[model_key] = pipeline
    
    # Déterminer la meilleure combinaison (modèle BERT + stratégie + classifieur)
    best_score = 0
    best_config = None
    
    print("\n=== Résumé des résultats ===")
    summary_data = []
    
    for model_key, result in results.items():
        model, strategy = model_key.split('_')
        best_clf = result['best_classifier']
        clf_scores = result['classifier_results'][best_clf]
        
        # Score combiné (recall + f1 pour classe 1)
        score = (clf_scores['f1_class1'])
        
        summary_data.append({
            'Modèle BERT': model,
            'Stratégie': strategy,
            'Classifieur': best_clf,
            'Recall (classe 1)': clf_scores['recall_class1'],
            'F1 (classe 1)': clf_scores['f1_class1'],
            'Score': score
        })
        
        if score > best_score:
            best_score = score
            best_config = model_key
    
    # Afficher un résumé des résultats
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.sort_values('Score', ascending=False))
    
    if best_config:
        print(f"\nMeilleure configuration: {best_config} avec {results[best_config]['best_classifier']}")
        best_model_key = best_config
        best_clf_name = results[best_config]['best_classifier']
        
        # Préparer les données pour l'optimisation
        model_name, embedding_strategy = best_model_key.split('_')
        embedder = BERTEmbedder(model_name=model_name, max_length=128, embedding_strategy=embedding_strategy)
        
        # Transformer les données
        X_train = twenty_train.processed_text.tolist() if hasattr(twenty_train, 'processed_text') else twenty_train.tolist()
        X_test = twenty_test.processed_text.tolist() if hasattr(twenty_test, 'processed_text') else twenty_test.tolist()
        y_train = twenty_train.target if hasattr(twenty_train, 'target') else twenty_train
        y_test = twenty_test.target if hasattr(twenty_test, 'target') else twenty_test
        
        print("\nCréation des embeddings pour l'optimisation...")
        X_train_embedded = embedder.fit_transform(X_train)
        X_test_embedded = embedder.transform(X_test)
        
        # Optimiser le meilleur classifieur
        best_model = optimize_best_classifier(
            best_clf_name, X_train_embedded, y_train, X_test_embedded, y_test
        )
        
        # Créer la pipeline finale
        final_pipeline = Pipeline([
            ('embedder', embedder),
            ('classifier', best_model)
        ])

        # Exporter le modèle
        print("\nExportation du modèle au format pkl...")
        with open(f'bert_{model_name.replace("-", "_")}_{embedding_strategy}_{best_clf_name}.pkl', 'wb') as f:
            pickle.dump(final_pipeline, f)
        print(f"Modèle exporté: bert_{model_name.replace('-', '_')}_{embedding_strategy}_{best_clf_name}.pkl")
        
        return final_pipeline, embedder
    
    return None, None