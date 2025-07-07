# 🌱 Medicinal Plant Classifier – Classificateur de Plantes Médicinales
Projet universitaire développé par Abderrahman AJINOU (Université Paris Cité, N° Étudiant : 22116322)

---

## 🤖 Pourquoi ce projet est-il un vrai projet d'intelligence artificielle ?
Ce projet n'est pas juste un formulaire ou un site web : il s'agit d'une application d'intelligence artificielle appliquée à la botanique. Voici pourquoi :

- **Utilisation du machine learning** : le cœur du projet repose sur un modèle d'apprentissage automatique (RandomForest, etc.) pour classer les plantes selon leurs caractéristiques.
- **Manipulation de données réelles** : le projet utilise un jeu de données de plantes, avec des attributs réels (type de feuille, habitat, couleur de la fleur, etc.).
- **Prétraitement et évaluation** : le code inclut l'encodage, la normalisation, la séparation train/test, et le calcul de métriques (accuracy, recall, F1-score, courbe ROC).
- **Interface interactive** : l'utilisateur peut saisir les caractéristiques d'une plante et obtenir une prédiction en temps réel, avec explications et visualisations.
- **Respect des standards IA** : le projet suit les bonnes pratiques du machine learning (prétraitement, modularité, visualisation, évaluation).

En résumé : ce classificateur est un vrai projet d'IA (machine learning appliqué à la botanique), et non un simple projet d'algorithmique ou d'interface. Il peut être présenté comme tel dans un contexte académique.

---

## 🚀 Présentation
Medicinal Plant Classifier est une application web qui prédit si une plante est médicinale ou non, à partir de ses caractéristiques. Le projet est développé en Python 3.10 avec Flask pour l’interface web, et propose un design moderne inspiré de la nature (palette verte, images, icônes SVG, Tailwind).

---

## 🗂️ Structure du projet
```
Medicinal Plant Classifier/
├── app.py                  # Point d'entrée Flask
├── requirements.txt        # Dépendances Python
├── model/
│   └── plant_classifier.py # Logique IA (RandomForest, prétraitement, évaluation)
├── controller/
│   └── routes.py           # Routes Flask (MVC)
├── templates/
│   └── index.html          # Page principale (formulaire, résultats dynamiques)
├── view/
│   └── results.html        # (optionnel) Page de résultats
├── static/
│   ├── images/             # Images, icônes Lucide, visualisations
│   ├── css/
│   │   └── style.css       # Styles personnalisés (optionnel)
│   └── visualizations/     # Graphiques générés (ROC, confusion, distribution)
├── visualizations/
│   ├── roc_curve.png
│   ├── confusion_matrix.png
│   └── class_distribution.png
├── data/
│   ├── plants.csv          # Jeu de données principal
│   └── plants_extended.csv # (optionnel)
├── devbook.md              # Suivi du développement (étapes, TODO)
├── generate_images.py      # Script pour générer des images de visualisation
└── Tache.txt               # Notes et tâches diverses
```

---

## 🛠️ Installation et environnement
### 1. Via Miniconda (recommandé)
```bash
# Installer Miniconda si besoin : https://docs.conda.io/en/latest/miniconda.html
source ~/miniconda3/etc/profile.d/conda.sh
conda create -n plant-ia python=3.10
conda activate plant-ia
pip install -r requirements.txt
```
### 2. Via venv (si Python 3.10+ installé)
```bash
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 📦 Dépendances principales
- Flask==2.3.3
- pandas==2.1.4
- scikit-learn==1.3.2
- matplotlib==3.8.2
- numpy==1.26.3
- seaborn==0.13.0

Toutes les versions sont précisées dans `requirements.txt`.

---

## ⚙️ Fonctionnement & points d’entrée
- `app.py` : Lance l’interface web Flask (formulaire, prédiction, résultats dynamiques)
- `model/plant_classifier.py` : Logique IA (prétraitement, entraînement, prédiction, évaluation)
- `controller/routes.py` : Routes Flask (MVC)
- `templates/index.html` : Formulaire principal, affichage dynamique des résultats
- `static/images/` : Images, icônes, visualisations

---

## 📊 Données
- `data/plants.csv` : Jeu de données principal (caractéristiques des plantes)
- `data/plants_extended.csv` : Variante ou extension du dataset

---

## 🌐 Utilisation
1. Lancer l’application web
   ```bash
   conda activate plant-ia  # ou source venv/bin/activate
   python app.py
   ```
2. Accéder à l’interface : http://localhost:5000
3. Remplir le formulaire avec les caractéristiques de la plante
4. Cliquer sur « Analyser »
5. Obtenir la prédiction (médicinale ou non), les explications et les métriques

---

## 📈 Visualisations
Des graphiques sont générés automatiquement pour analyser le modèle :
- `static/images/class_distribution_train.png` : Distribution des classes (train)
- `static/images/confusion_matrix.png` : Matrice de confusion
- `static/images/roc_curve.png` : Courbe ROC
- `static/images/feature_importance.png` : Importance des features

Pour générer ou regénérer les graphiques :
```bash
python generate_images.py
```

---

## 🧪 Tests & vérifications
- Vérifier que l’application se lance sans erreur
- Tester plusieurs combinaisons dans le formulaire
- Les métriques (accuracy, recall, F1-score) s’affichent après chaque prédiction

---

## 📝 Bonnes pratiques & conseils d’étudiant
- **Ne versionnez pas** les fichiers volumineux ou sensibles (`.gitignore` déjà configuré)
- **Gardez l’environnement isolé** (conda ou venv)
- **Commentez votre code** pour mieux comprendre plus tard
- **Utilisez le devbook.md** pour suivre l’avancement et ne rien oublier
- **N’hésitez pas à demander de l’aide** si vous bloquez sur un bug ou une question IA

---

## 👤 Auteur
Abderrahman AJINOU  
Étudiant en 2ᵉ année licence Informatique Générale  
Université Paris Cité, Campus Grand Moulin  
Mail : abderrahman.ajinou@etu.u-paris.fr

---

## 📚 Ressources utiles
- [Documentation Flask](https://flask.palletsprojects.com/)
- [Documentation scikit-learn](https://scikit-learn.org/)
- [Documentation pandas](https://pandas.pydata.org/)
- [Documentation matplotlib](https://matplotlib.org/)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

🏆 Pour toute question ou suggestion, n’hésitez pas à me contacter !
