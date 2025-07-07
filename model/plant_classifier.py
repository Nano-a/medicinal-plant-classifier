import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, classification_report
import os
import matplotlib
matplotlib.use('Agg')  # Utiliser le backend non interactif
import matplotlib.pyplot as plt
import seaborn as sns


# Documentation du modèle
"""
Ce modèle utilise un RandomForestClassifier pour prédire si une plante est médicinale.
Le RandomForest est un algorithme plus complexe que la Régression Logistique, mais reste accessible.
Il fonctionne en créant plusieurs arbres de décision et en combinant leurs prédictions.

Avantages du RandomForest :
1. Plus robuste aux données bruitées
2. Gestion automatique des interactions entre variables
3. Pas besoin de normalisation des données
4. Meilleure performance que la Régression Logistique sur des données non linéaires
"""

class PlantClassifier:
    def __init__(self, data_path='data/plants.csv'):
        self.data_path = data_path
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = None  # Stocker les noms des caractéristiques
        self._initialize_model()  # Initialiser le modèle dès le début
        
        # Charger les données pour initialiser les encodeurs
        try:
            self._initialize_encoders()
            self._initialize_scaler()
            
            # Entraîner le modèle initial
            self.train()
        except Exception as e:
            print(f"Erreur lors de l'initialisation: {str(e)}")

    def _initialize_encoders(self):
        """Initialise les encodeurs avec les valeurs possibles"""
        try:
            # Charger les données pour obtenir les valeurs uniques
            df = pd.read_csv(self.data_path)
            
            # Initialiser les encodeurs pour chaque colonne catégorielle
            for column in df.select_dtypes(include=['object']).columns:
                if column not in ['medicinal']:
                    self.label_encoders[column] = LabelEncoder()
                    # Ajuster l'encodeur aux données
                    self.label_encoders[column].fit(df[column])
            
            # Sauvegarder les valeurs uniques pour chaque encodeur
            self.encoder_values = {}
            for col, encoder in self.label_encoders.items():
                self.encoder_values[col] = list(encoder.classes_)
        except Exception as e:
            print(f"Erreur lors de l'initialisation des encodeurs: {str(e)}")
            raise

    def _initialize_scaler(self):
        """Initialise le scaler"""
        self.scaler = StandardScaler()

    def _initialize_model(self):
        """Initialise le modèle RandomForest"""
        try:
            # Vérifier si le modèle est déjà initialisé
            if self.model is not None:
                print("Modèle déjà initialisé")
                return
            
            # Initialiser le modèle avec des paramètres optimisés
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
                class_weight='balanced'
            )
            
            # Vérifier que le modèle a été correctement initialisé
            if self.model is None:
                raise ValueError("Impossible d'initialiser le modèle RandomForest")
            
        except Exception as e:
            print(f"Erreur lors de l'initialisation du modèle: {str(e)}")
            raise ValueError(f"Erreur lors de l'initialisation du modèle: {str(e)}")

    def predict(self, leaf_type, habitat, flower_color, height, season='summer', leaf_size='medium', stem_texture='smooth'):
        """Fait une prédiction pour une plante donnée
        
        Args:
            leaf_type (str): Type de feuille (simple/compound)
            habitat (str): Habitat de la plante
            flower_color (str): Couleur de la fleur
            height (float): Hauteur de la plante en cm
            season (str, optional): Saison. Defaults to 'summer'.
            leaf_size (str, optional): Taille des feuilles. Defaults to 'medium'.
            stem_texture (str, optional): Texture du tronc. Defaults to 'smooth'.
            
        Returns:
            dict: Résultat de la prédiction et explication
        """
        try:
            # Vérifier que le modèle est entraîné
            if self.model is None:
                raise ValueError("Le modèle n'est pas entraîné")
            
            # Créer un DataFrame avec les noms de colonnes dans le bon ordre
            sample = pd.DataFrame({
                'leaf_type': [leaf_type],
                'habitat': [habitat],
                'flower_color': [flower_color],
                'height': [float(height)],
                'season': [season],
                'leaf_size': [leaf_size],
                'stem_texture': [stem_texture]
            }, columns=self.feature_names)
            
            # Encoder les variables catégorielles
            for column in sample.select_dtypes(include=['object']).columns:
                if column in self.label_encoders:
                    sample[column] = self.label_encoders[column].transform(sample[column])
            
            # Appliquer le scaler
            sample_scaled = self.scaler.transform(sample)
            
            # Faire la prédiction
            prediction = self.model.predict(sample_scaled)[0]
            probability = self.model.predict_proba(sample_scaled)[0][1]
            
            # Calculer l'importance des caractéristiques pour l'explication
            importance_dict = {}
            for feature, importance in zip(self.feature_names, self.model.feature_importances_):
                importance_dict[feature] = importance
            
            # Trier et formater l'explication
            explanation = []
            for feature, importance in sorted(importance_dict.items(), key=lambda x: x[1], reverse=True):
                explanation.append(f"- {feature}: {importance:.2%}")
            
            # Obtenir les métriques du modèle
            metrics = self._get_model_metrics()
            
            return {
                'result': 'Médicinale' if prediction == 1 else 'Non médicinale',
                'probability': float(probability),
                'explanation': explanation,
                'metrics': metrics
            }
        except Exception as e:
            print(f"Erreur lors de la prédiction: {str(e)}")
            raise

    def load_and_prepare_data(self):
        """Charge et prépare les données"""
        try:
            # Charger les données
            df = pd.read_csv(self.data_path)
            
            # Vérifier que les colonnes nécessaires existent
            required_columns = ['leaf_type', 'habitat', 'flower_color', 'height', 'season', 'leaf_size', 'stem_texture', 'medicinal']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"Colonnes manquantes: {set(required_columns) - set(df.columns)}")
            
            # Stocker les noms des caractéristiques avant l'encodage
            self.feature_names = df.drop(['medicinal'], axis=1).columns.tolist()
            
            # Encoder les variables catégorielles
            for column in df.select_dtypes(include=['object']).columns:
                if column in self.label_encoders:
                    df[column] = self.label_encoders[column].transform(df[column])
            
            # Séparer les caractéristiques et la cible
            X = df.drop(['medicinal'], axis=1)
            y = df['medicinal'].map({'yes': 1, 'no': 0})
            
            # Normaliser les données
            X_scaled = self.scaler.fit_transform(X)
            
            return X_scaled, y
            
        except Exception as e:
            print(f"Erreur lors du chargement des données: {str(e)}")
            raise



    def _create_visualizations(self, X_train, y_train, X_test, y_test, y_pred):
        """Crée les visualisations du modèle"""
        try:
            # Créer le dossier des images s'il n'existe pas
            os.makedirs('static/images', exist_ok=True)
            
            # 1. Matrice de confusion
            try:
                plt.figure(figsize=(8, 6))
                cm = confusion_matrix(y_test, y_pred)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                           xticklabels=['Non médicinale', 'Médicinale'],
                           yticklabels=['Non médicinale', 'Médicinale'])
                plt.title('Matrice de confusion')
                plt.xlabel('Prédiction')
                plt.ylabel('Vraie valeur')
                plt.savefig('static/images/confusion_matrix.png')
                plt.close()
            except Exception as e:
                print(f"Erreur lors de la création de la matrice de confusion: {str(e)}")

            # 2. Distribution des classes
            try:
                plt.figure(figsize=(8, 6))
                sns.countplot(x=y_train, palette='coolwarm')
                plt.title('Distribution des classes')
                plt.xticks([0, 1], ['Non médicinale', 'Médicinale'])
                plt.savefig('static/images/class_distribution.png')
                plt.close()
            except Exception as e:
                print(f"Erreur lors de la création de la distribution des classes: {str(e)}")

            # 3. Importance des caractéristiques
            try:
                plt.figure(figsize=(10, 6))
                # Créer un DataFrame pour les importances des caractéristiques
                feature_importances_df = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': self.model.feature_importances_
                })
                
                # Trier et afficher les 10 caractéristiques les plus importantes
                feature_importances_df = feature_importances_df.sort_values('importance', ascending=False)
                sns.barplot(data=feature_importances_df.head(10), 
                           x='importance', y='feature',
                           palette='viridis')
                plt.title('Importance des caractéristiques')
                plt.xlabel('Importance')
                plt.savefig('static/images/feature_importances.png')
                plt.close()
            except Exception as e:
                print(f"Erreur lors de la création de l'importance des caractéristiques: {str(e)}")

            # 4. Courbe ROC
            try:
                plt.figure(figsize=(8, 6))
                fpr, tpr, _ = roc_curve(y_test, self.model.predict_proba(X_test)[:, 1])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic')
                plt.legend(loc="lower right")
                plt.savefig('static/images/roc_curve.png')
                plt.close()
            except Exception as e:
                print(f"Erreur lors de la création de la courbe ROC: {str(e)}")

        except Exception as e:
            print(f"Erreur générale lors de la création des visualisations: {str(e)}")
            raise

    def _get_model_metrics(self):
        """Obtient les métriques du modèle"""
        try:
            # Vérifier que le modèle est entraîné
            if self.model is None:
                raise ValueError("Le modèle n'est pas entraîné")
            
            # Prédire sur les données de test
            y_pred = self.model.predict(self.X_test)
            y_prob = self.model.predict_proba(self.X_test)[:, 1]
            
            # Calculer les métriques
            metrics = {
                'accuracy': accuracy_score(self.y_test, y_pred),
                'precision': precision_score(self.y_test, y_pred, pos_label=1),
                'recall': recall_score(self.y_test, y_pred, pos_label=1),
                'f1_score': f1_score(self.y_test, y_pred, pos_label=1),
                'n_samples': len(self.y_test),
                'n_medicinal': sum(self.y_test),
                'n_non_medicinal': len(self.y_test) - sum(self.y_test),
                'medicinal_ratio': sum(self.y_test) / len(self.y_test),
                'confusion_matrix': confusion_matrix(self.y_test, y_pred).tolist(),
                'roc_auc': roc_auc_score(self.y_test, y_prob)
            }
            
            # Ajouter les métriques de performance détaillées
            cm = confusion_matrix(self.y_test, self.y_pred)
            metrics['confusion_matrix'] = cm.tolist()
            
            # Calculer la courbe ROC et AUC
            y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            metrics['roc_auc'] = float(auc(fpr, tpr))
            
            return metrics
            
        except Exception as e:
            print(f"Erreur lors de l'obtention des métriques: {str(e)}")
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'n_samples': 0,
                'n_medicinal': 0,
                'n_non_medicinal': 0,
                'medicinal_ratio': 0.0,
                'confusion_matrix': [[0, 0], [0, 0]],
                'roc_auc': 0.0
            }

    def train(self):
        """Entraîne le modèle et génère les visualisations
        
        Étapes :
        1. Préparation des données
        2. Entraînement du modèle
        3. Évaluation des performances
        """
        
        try:
            # Charger et préparer les données
            X, y = self.load_and_prepare_data()
            
            # Vérifier que nous avons assez de données pour le train/test
            if len(X) < 5:  # Au moins 5 échantillons pour un split 80/20
                raise ValueError("Pas assez de données pour le train/test split")
            
            # Séparer les données en train et test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Entraîner le modèle
            self.model.fit(X_train, y_train)
            
            # Faire des prédictions
            self.y_pred = self.model.predict(X_test)
            
            # Stocker les données de test
            self.X_test = X_test
            self.y_test = y_test
            
            # Obtenir les métriques du modèle
            metrics = self._get_model_metrics()
            
            # Ajouter la distribution des classes dans les jeux de données
            metrics['train_medicinal_ratio'] = float(sum(y_train) / len(y_train))
            metrics['test_medicinal_ratio'] = float(sum(y_test) / len(y_test))

            # Ajouter les métriques de performance détaillées
            metrics['confusion_matrix'] = confusion_matrix(y_test, self.y_pred).tolist()
            metrics['class_report'] = classification_report(y_test, self.y_pred, output_dict=True)
            
            return metrics
            
        except Exception as e:
            print(f"Erreur lors de l'entraînement: {str(e)}")
            # Retourner des métriques par défaut en cas d'erreur
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'roc_auc': 0.0,
                'n_samples': 0,
                'n_train': 0,
                'n_test': 0,
                'n_medicinal': 0,
                'n_non_medicinal': 0,
                'train_medicinal_ratio': 0.0,
                'test_medicinal_ratio': 0.0,
                'confusion_matrix': [[0, 0], [0, 0]],
                'class_report': {
                    '0': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0},
                    '1': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0},
                    'accuracy': 0.0,
                    'macro avg': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0},
                    'weighted avg': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0}
                }
            }