#!/bin/bash

# Vérifier si le répertoire de données existe
if [ ! -d "/app/data" ]; then
    mkdir -p /app/data
    echo "Création du répertoire /app/data"
fi

# Vérifier si le fichier tweets.csv existe
if [ ! -f "/app/data/tweets.csv" ]; then
    echo "id,keyword,location,text,target,prediction,probability,timestamp" > /app/data/tweets.csv
    echo "Création du fichier tweets.csv avec en-têtes"
fi

# Vérifier si le répertoire du modèle existe
if [ ! -d "/app/model" ]; then
    mkdir -p /app/model
    echo "Création du répertoire /app/model"
fi

# Vérifier si le modèle existe
if [ ! -f "/app/model/bert_model.pkl" ]; then
    echo "AVERTISSEMENT: Le modèle n'existe pas à /app/model/bert_model.pkl"
    echo "Veuillez vous assurer que le modèle est disponible avant d'utiliser l'application"
fi

# Démarrer Streamlit
echo "Démarrage de l'application Streamlit..."
streamlit run app.py