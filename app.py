import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import plotly.express as px
from model.Module import BERTEmbedder


# Configuration de la page
st.set_page_config(
    page_title="Détecteur de tweets annonciateurs de catastrophes",
    page_icon="",
    layout="wide"
)

# Titre et description
st.title("Détecteur de tweets annonciateurs de catastrophes")
st.markdown("""
Cette application permet de classifier des tweets pour détecter s'ils rapportent des incidents réels ou non.
Le modèle est basé sur BERT et a été entraîné pour identifier des tweets liés à des catastrophes ou incidents.
""")

# Fonction pour charger le modèle
@st.cache_resource
def load_model():
    model_path = "model/bert_model.pkl"
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle: {e}")
        return None

# Fonction pour faire des prédictions
def predict_tweet(tweet_text, model):
    try:
        # Le modèle est une pipeline qui inclut l'embedder et le classifieur
        prediction = model.predict([tweet_text])[0]
        probability = model.predict_proba([tweet_text])[0][1]  # Probabilité de la classe 1
        return prediction, probability
    except Exception as e:
        st.error(f"Erreur lors de la prédiction: {e}")
        return None, None

# Fonction pour charger les données
@st.cache_data
def load_data():
    try:
        # Vérifier si le fichier existe
        data_path = "model/data/tweets.csv"
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            return df
        else:
            # Si le fichier n'existe pas, créer un DataFrame vide avec les colonnes nécessaires
            df = pd.DataFrame(columns=["id", "keyword", "location", "text", "target", "prediction", "probability", "timestamp"])
            df.to_csv(data_path, index=False)
            return df
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {e}")
        return pd.DataFrame()

# Chargement du modèle et des données
model = load_model()
df_tweets = load_data()

# Sidebar pour la navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choisir une page", ["Tableau de bord", "Analyser un tweet", "Ajouter des données"])

# Page: Tableau de bord
if page == "Tableau de bord":
    st.header("Tableau de bord")
    
    # Affichage des statistiques
    if not df_tweets.empty:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_tweets = len(df_tweets)
            st.metric("Total de tweets", total_tweets)
        
        with col2:
            if "target" in df_tweets.columns:
                real_incidents = df_tweets[df_tweets["target"] == 1].shape[0]
                st.metric("Incidents réels", real_incidents)
            else:
                st.metric("Incidents réels", 0)
        
        with col3:
            if "target" in df_tweets.columns:
                not_incidents = df_tweets[df_tweets["target"] == 0].shape[0]
                st.metric("Non-incidents", not_incidents)
            else:
                st.metric("Non-incidents", 0)
        
        # Graphiques
        if "target" in df_tweets.columns and len(df_tweets) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                # Histogramme des incidents par mot-clé
                if "keyword" in df_tweets.columns:
                    keyword_counts = df_tweets.groupby(['keyword', 'target']).size().reset_index(name='count')
                    if not keyword_counts.empty:
                        fig = px.bar(keyword_counts, x='keyword', y='count', color='target',
                                    labels={'target': 'Type', 'count': 'Nombre', 'keyword': 'Mot-clé'},
                                    title="Tweets par mot-clé",
                                    color_discrete_map={0: "#3498db", 1: "#e74c3c"})
                        st.plotly_chart(fig)
            
            with col2:
                # Distribution des prédictions vs réalité
                if "prediction" in df_tweets.columns and "target" in df_tweets.columns:
                    confusion_data = pd.crosstab(df_tweets['target'], df_tweets['prediction'], 
                                            rownames=['Réel'], colnames=['Prédit'])
                    fig = px.imshow(confusion_data, 
                                    labels=dict(x="Prédit", y="Réel", color="Nombre"),
                                    x=['Non-incident (0)', 'Incident (1)'],
                                    y=['Non-incident (0)', 'Incident (1)'],
                                    text_auto=True,
                                    color_continuous_scale='Blues')
                    st.plotly_chart(fig)
        
        # Affichage des tweets avec filtres
        st.subheader("Tweets analysés")
        
        # Filtres
        col1, col2, col3 = st.columns(3)
        with col1:
            if "keyword" in df_tweets.columns:
                keywords = ['Tous'] + sorted(df_tweets['keyword'].unique().tolist())
                keyword_filter = st.selectbox("Filtrer par mot-clé", keywords)
        
        with col2:
            if "target" in df_tweets.columns:
                target_options = {
                    'Tous': None,
                    'Incident (1)': 1,
                    'Non-incident (0)': 0
                }
                target_filter = st.selectbox("Filtrer par type", list(target_options.keys()))
        
        with col3:
            if "location" in df_tweets.columns:
                locations = ['Tous'] + sorted([loc for loc in df_tweets['location'].unique() if pd.notna(loc)])
                location_filter = st.selectbox("Filtrer par lieu", locations)
        
        # Application des filtres
        filtered_df = df_tweets.copy()
        
        if "keyword" in df_tweets.columns and keyword_filter != 'Tous':
            filtered_df = filtered_df[filtered_df['keyword'] == keyword_filter]
        
        if "target" in df_tweets.columns and target_filter != 'Tous':
            filtered_df = filtered_df[filtered_df['target'] == target_options[target_filter]]
        
        if "location" in df_tweets.columns and location_filter != 'Tous':
            filtered_df = filtered_df[filtered_df['location'] == location_filter]
        
        # Affichage du tableau filtré
        if len(filtered_df) > 0:
            display_cols = ['text', 'keyword', 'location', 'target', 'prediction', 'probability']
            display_cols = [col for col in display_cols if col in filtered_df.columns]
            st.dataframe(filtered_df[display_cols], use_container_width=True)
        else:
            st.info("Aucun tweet ne correspond aux filtres sélectionnés.")
    else:
        st.info("Aucune donnée disponible. Veuillez ajouter des tweets via l'onglet 'Ajouter des données'.")

# Page: Analyser un tweet
elif page == "Analyser un tweet":
    st.header("Analyser un tweet")
    
    # Zone de texte pour entrer un tweet
    tweet_text = st.text_area("Entrez le texte du tweet à analyser", height=100)
    
    # Informations supplémentaires
    col1, col2 = st.columns(2)
    with col1:
        keyword = st.text_input("Mot-clé (optionnel)")
    with col2:
        location = st.text_input("Localisation (optionnel)")
    
    # Bouton pour analyser
    if st.button("Analyser"):
        if tweet_text:
            if model:
                # Prédiction
                prediction, probability = predict_tweet(tweet_text, model)
                
                if prediction is not None:
                    # Affichage du résultat
                    st.subheader("Résultat de l'analyse")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if prediction == 1:
                            st.error("Ce tweet rapporte probablement une catastrophe")
                        else:
                            st.success("Ce tweet ne rapporte probablement pas une catastrophe")
                    
                    with col2:
                        st.metric("La probabilité pour que ce tweet ce rapporte a une catastrophe est de", f"{probability:.2%}")
                    
                    # Option pour ajouter à la base de données
                    if st.button("Ajouter ce tweet à la base de données"):
                        # Générer un ID simple basé sur l'horodatage
                        tweet_id = len(df_tweets)
                        
                        # Créer une nouvelle ligne
                        new_row = {
                            "id": tweet_id,
                            "keyword": keyword,
                            "location": location,
                            "text": tweet_text,
                            "prediction": int(prediction),
                            "probability": float(probability),
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        
                        # Ajouter la ligne au DataFrame
                        df_tweets_new = pd.concat([df_tweets, pd.DataFrame([new_row])], ignore_index=True)
                        
                        # Sauvegarder le DataFrame mis à jour
                        df_tweets_new.to_csv("data/tweets.csv", index=False)
                        st.success("Tweet ajouté à la base de données avec succès!")
                        
                        # Mettre à jour le DataFrame en cache
                        st.cache_data.clear()
            else:
                st.error("Le modèle n'a pas pu être chargé. Veuillez vérifier le fichier du modèle.")
        else:
            st.warning("Veuillez entrer un texte à analyser.")

# Page: Ajouter des données
elif page == "Ajouter des données":
    st.header("Ajouter des données")
    
    # Option 1: Ajouter un tweet manuellement
    st.subheader("Option 1: Ajouter un tweet manuellement")
    
    # Formulaire pour ajouter un tweet
    with st.form(key="add_tweet_form"):
        tweet_text = st.text_area("Texte du tweet", height=100)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            keyword = st.text_input("Mot-clé")
        with col2:
            location = st.text_input("Localisation")
        with col3:
            target = st.selectbox("Type", [("Non-incident", 0), ("Incident", 1)], format_func=lambda x: x[0])
        
        submit_button = st.form_submit_button(label="Ajouter")
    
    if submit_button:
        if tweet_text:
            # Prédiction automatique si le modèle est disponible
            prediction, probability = (None, None)
            if model:
                prediction, probability = predict_tweet(tweet_text, model)
            
            # Générer un ID simple
            tweet_id = len(df_tweets)
            
            # Créer une nouvelle ligne
            new_row = {
                "id": tweet_id,
                "keyword": keyword,
                "location": location,
                "text": tweet_text,
                "target": target[1]
            }
            
            # Ajouter la prédiction si disponible
            if prediction is not None:
                new_row["prediction"] = int(prediction)
                new_row["probability"] = float(probability)
            
            # Ajouter l'horodatage
            new_row["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Ajouter la ligne au DataFrame
            df_tweets_new = pd.concat([df_tweets, pd.DataFrame([new_row])], ignore_index=True)
            
            # Sauvegarder le DataFrame mis à jour
            df_tweets_new.to_csv("data/tweets.csv", index=False)
            st.success("Tweet ajouté avec succès!")
            
            # Mettre à jour le DataFrame en cache
            st.cache_data.clear()
        else:
            st.warning("Veuillez entrer un texte pour le tweet.")
    
    # Option 2: Importer des données à partir d'un fichier CSV
    st.subheader("Option 2: Importer des données à partir d'un fichier CSV")
    
    uploaded_file = st.file_uploader("Choisir un fichier CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            # Lecture du fichier
            imported_df = pd.read_csv(uploaded_file)
            
            # Vérification des colonnes requises
            required_cols = ["text"]
            if not all(col in imported_df.columns for col in required_cols):
                st.error(f"Le fichier CSV doit contenir au moins la colonne: {', '.join(required_cols)}")
            else:
                # Prévisualisation des données
                st.write("Aperçu des données importées:")
                st.dataframe(imported_df.head())
                
                # Option pour ajouter les prédictions automatiquement
                add_predictions = st.checkbox("Ajouter des prédictions automatiquement", value=True)
                
                # Bouton pour confirmer l'importation
                if st.button("Importer ces données"):
                    # Ajout des prédictions
                    if add_predictions and model:
                        # Création d'une barre de progression
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Liste pour stocker les prédictions
                        predictions = []
                        probabilities = []
                        
                        # Calcul des prédictions
                        total_rows = len(imported_df)
                        for i, row in enumerate(imported_df['text']):
                            pred, prob = predict_tweet(row, model)
                            predictions.append(int(pred) if pred is not None else None)
                            probabilities.append(float(prob) if prob is not None else None)
                            
                            # Mise à jour de la barre de progression
                            progress = (i + 1) / total_rows
                            progress_bar.progress(progress)
                            status_text.text(f"Traitement: {i+1}/{total_rows} tweets ({progress:.1%})")
                        
                        # Ajout des colonnes de prédiction
                        imported_df['prediction'] = predictions
                        imported_df['probability'] = probabilities
                    
                    # Ajout des ID si nécessaire
                    if 'id' not in imported_df.columns:
                        start_id = len(df_tweets)
                        imported_df['id'] = range(start_id, start_id + len(imported_df))
                    
                    # Ajout de l'horodatage
                    imported_df['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Fusion des DataFrames
                    combined_df = pd.concat([df_tweets, imported_df], ignore_index=True)
                    
                    # Sauvegarde
                    combined_df.to_csv("model/data/tweets.csv", index=False)
                    st.success(f"{len(imported_df)} tweets importés avec succès!")
                    
                    # Mettre à jour le DataFrame en cache
                    st.cache_data.clear()
        
        except Exception as e:
            st.error(f"Erreur lors de l'importation: {e}")

# Pied de page
st.sidebar.markdown("---")
st.sidebar.markdown("© 2025 Détecteur de tweets annonciateurs de catastrophes")