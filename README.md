# Tweets_catastrophes
Détecter les tweets annonciateurs de catastrophes

Groupe 1:
Membres du groupe:
Boubacar KANE
Salimou ABDOULAYE HALIDOU
Birahime KAMARA
Parfait Jemmy Prodige NGOYI

# Détecteur de catastrophes par tweets

Cette application analyse des tweets pour détecter s'ils rapportent des catastrophes réels ou non, à l'aide d'un modèle basé sur BERT.

## Fonctionnalités

- **Tableau de bord** : Visualisation et filtrage des tweets avec leurs prédictions
- **Analyse de tweet** : Analyse de nouveaux tweets en temps réel
- **Ajout de données** : Ajout manuel de tweets ou import par lot via CSV

## Installation et lancement

### Avec Docker (recommandé)

1. Placez votre fichier modèle `bert_model.pkl` dans le dossier `model/`
2. Construisez et lancez le conteneur Docker :

```bash
docker-compose up -d
```

L'application sera accessible à l'adresse https://tweetscatastrophes-etzwg8ha9edt3qgjbx9vjj.streamlit.app/

### Sans Docker

1. Créez un environnement virtuel et installez les dépendances :

```bash
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. Placez votre fichier modèle `bert_model.pkl` dans le dossier `model/`
3. Lancez l'application :

```bash
streamlit run app.py
```

## Structure du projet

```
.
├── app.py                # Application Streamlit principale
├── data/                 # Dossier contenant les données
│   └── tweets.csv        # CSV des tweets analysés
├── model/                # Dossier contenant le modèle
│   └── bert_model.pkl    # Modèle BERT exporté
├── Dockerfile            # Configuration Docker
├── docker-compose.yml    # Configuration Docker Compose
├── requirements.txt      # Dépendances Python
├── start.sh              # Script de démarrage pour Docker
└── README.md             # Documentation
```

## Préparation du modèle

Le modèle utilisé est un pipeline sklearn contenant un BERTEmbedder et un classificateur. Il a été entraîné pour détecter si un tweet rapporte un incident réel ou non.

Assurez-vous que votre modèle exporté (`bert_model.pkl`) est placé dans le dossier `model/` avant de lancer l'application.

## Format des données

Le format attendu pour les tweets est le suivant :
- `id` : Identifiant unique
- `keyword` : Mot-clé associé au tweet
- `location` : Localisation mentionnée dans le tweet
- `text` : Texte du tweet
- `target` : Étiquette réelle (1 pour incident, 0 pour non-incident)
- `prediction` : Prédiction du modèle
- `probability` : Probabilité de la prédiction
- `timestamp` : Horodatage de l'analyse
