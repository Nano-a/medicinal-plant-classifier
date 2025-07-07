from flask import Blueprint, render_template, request, jsonify
from model.plant_classifier import PlantClassifier

# Initialiser le classifieur
plant_classifier = PlantClassifier()

# Créer le blueprint
main = Blueprint('main', __name__)

@main.route('/')
def index():
    """Route principale"""
    return render_template('index.html')

@main.route('/predict', methods=['POST'])
def predict():
    """Route de prédiction"""
    try:
        print("\n=== Début de la route predict ===")
        
        # Vérifier que le modèle est initialisé
        print(f"État du modèle: {'entraîné' if plant_classifier.model is not None else 'non entraîné'}")
        
        # Récupérer les données du formulaire HTML (POST classique)
        data = request.form
        print(f"\nDonnées reçues dans la requête:")
        print(data)
        
        # Vérifier que les données sont valides
        required_fields = ['leaf_type', 'habitat', 'flower_color', 'height', 'season', 'leaf_size', 'stem_texture']
        missing_fields = [field for field in required_fields if field not in data or not data.get(field)]
        
        if missing_fields:
            print(f"\nChamps manquants: {missing_fields}")
            return jsonify({
                'error': f'Champs manquants dans la requête: {", ".join(missing_fields)}'
            }), 400
        
        # Vérifier et convertir la hauteur
        try:
            height = float(data.get('height'))
            if height <= 0:
                raise ValueError('La hauteur doit être supérieure à 0')
        except (ValueError, TypeError) as e:
            print(f"\nErreur de validation de la hauteur: {str(e)}")
            return jsonify({
                'error': str(e)
            }), 400
        
        # Faire la prédiction avec tous les paramètres
        try:
            print("\nFaisant la prédiction avec tous les paramètres...")
            prediction = plant_classifier.predict(
                data.get('leaf_type'),
                data.get('habitat'),
                data.get('flower_color'),
                height,
                data.get('season'),
                data.get('leaf_size'),
                data.get('stem_texture')
            )
            
            if prediction is None:
                raise ValueError('La prédiction a retourné None')
                
            print(f"\nRésultat de la prédiction:")
            print(prediction)
            
            # Vérifier et formater les métriques
            metrics = prediction.get('metrics', {})
            
            if metrics:
                print(f"\nMétriques reçues:")
                print(metrics)
                
                formatted_metrics = {}
                # Métriques de base (en pourcentage)
                for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                    if metric in metrics:
                        try:
                            value = float(metrics[metric])
                            formatted_metrics[metric] = f"{value * 100:.2f}%"
                        except (ValueError, TypeError):
                            formatted_metrics[metric] = "0.00%"
                
                # Métriques de comptage (en nombres entiers)
                for metric in ['n_samples', 'n_train', 'n_test', 'n_medicinal', 'n_non_medicinal']:
                    if metric in metrics:
                        try:
                            value = int(metrics[metric])
                            formatted_metrics[metric] = str(value)
                        except (ValueError, TypeError):
                            formatted_metrics[metric] = "0"
                
                # Métriques de ratio (en pourcentage)
                for metric in ['train_medicinal_ratio', 'test_medicinal_ratio']:
                    if metric in metrics:
                        try:
                            value = float(metrics[metric])
                            formatted_metrics[metric] = f"{value * 100:.2f}%"
                        except (ValueError, TypeError):
                            formatted_metrics[metric] = "0.00%"
            else:
                print("\nAucune métrique reçue")
                formatted_metrics = {
                    'accuracy': '0.00%',
                    'precision': '0.00%',
                    'recall': '0.00%',
                    'f1_score': '0.00%',
                    'n_samples': '0',
                    'n_train': '0',
                    'n_test': '0',
                    'n_medicinal': '0',
                    'n_non_medicinal': '0',
                    'train_medicinal_ratio': '0.00%',
                    'test_medicinal_ratio': '0.00%'
                }
            
            return jsonify({
                'result': prediction['result'],
                'explanation': prediction['explanation'],
                'metrics': formatted_metrics
            })
            
        except ValueError as e:
            print(f"\nErreur de validation lors de la prédiction: {str(e)}")
            return jsonify({
                'error': str(e)
            }), 400
            
        except Exception as e:
            print(f"\nErreur inattendue lors de la prédiction: {str(e)}")
            return jsonify({
                'error': 'Une erreur est survenue lors du traitement de la requête'
            }), 500
            
    except Exception as e:
        print(f"\nErreur dans la route predict: {str(e)}")
        return jsonify({
            'error': 'Une erreur est survenue lors du traitement de la requête'
        }), 500
