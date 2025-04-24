# Utiliser une image Python officielle
FROM python:3.9-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers de dépendances
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste de l'application
COPY . .

# Créer les répertoires nécessaires
RUN mkdir -p /app/data /app/model

# Rendre le script de démarrage exécutable
RUN chmod +x /app/start.sh

# Exposer le port utilisé par Streamlit
EXPOSE 8501

# Commande de démarrage
CMD ["/app/start.sh"]