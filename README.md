## Heaatseeker AI
![HeaatseekerAI Logo](./image.png)

## Introduction:
Depuis 2020, j'achète et revends des sneakers partout dans le monde notamment aux Etats-Unis et en Italie. 

Cependant, un des problème majeur de ce business est la contrefaçon. 
EN 2023, le marché mondial des sneakers contrefaits serait estimé à environ 600 milliards de $. 
Pour essayer de pallier ce problème, j'ai décidé d'utiliser mes compétences en informatique pour voir à quel point le projet pouvait être efficace. 
L'idée m'est venue grâce à un projet scolaire sur la reconnaissance de vêtements grâce à un CNN.

Je me suis alors demandé si on peut entraîner un modèle à reconnaître une jupe plutôt qu'une robe, est-ce que l'on peut entrainer un modèle a identifié une vraie sneaker d'une fausse ? 

## Description
Le projet HeaatseekerAI vise à authentifier des paires de sneakers de manière automatique et rapide. 
Le projet est divisé en 3 parties majeures:
- Collecte et Nettoyage d'images
- Entraînement du CNN
- Prédiction de l'authentification d'une paire
  
## Prérequis
Python 3.13

## Installation et exécution
1. Cloner le dépôt :
    ```bash
    git clone <url-du-repo>
    cd <nom-du-repo>
    ```
2. Créer et activer un environnement virtuel :
    - Windows :
      ```powershell
      python -m venv EnvVirtuel
      .\EnvVirtuel\Scripts\Activate.ps1
      ```
3. Installer les dépendances :
    ```bash
    pip install -r requirements.txt
    ```
4. Collecter les données (Paires Authentiques/ Paires Fausses)
5. Entraîner le modèle avec les données:  
    ```bash
    python EntrainementCNN.py
    ```
6. Une fois le modèle entraîné, le fichier PredictionLegit permet déterminer l'authenticité des paires dans le dossier Paires_Test:
    ```bash
    python PredictionLegit.py
    ```
Sortie attendue:
```bash
L'authentification de l'article article_10 a une confiance moyenne de 40.14%.
L'article semble LEGIT mais la confiance est faible.
Veuillez envoyer plus d'images pour une meilleure évaluation.
Article article_11: FAKE (Legit: 0, Fake: 5, Confiance moyenne: 59.33%)
Article article_12: FAKE (Legit: 0, Fake: 5, Confiance moyenne: 79.25%)
```
