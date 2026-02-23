## Heaatseeker AI
![Solution NQueens Problem](./image.png)

## Introduction:
Depuis 2020, j'achète et revends des sneakers partout dans le monde notamment aux Etats-Unis et en Italie. 
Cependant, un des problème majeur de ce business est la contrefaçon. 
EN 2023, le marché mondial des sneakers contrefaits serait estimé à environ 600 milliards de $. 
Pour essayer de pallier ce problème, j'ai décidé d'utiliser mes compétences en informatique pour voir à quel point le projet pouvait être efficace. 
L'idée m'est venue grâce à un projet scolaire sur la reconnaissance de vêtements grâce à un CNN.
Je me suis alors demandé si on peut entraîner un modèle à reconnaître une jupe plutôt qu'une robe, est-ce que l'on peut entrainer un modèle a identifié une vraie sneaker d'une fausse ? 

## Fonctionnement 
- Représentation des positions sur l'échiquier sous forme de paires de coordonnées (x, y)
- Détection des conflits entre les reines (même ligne, colonne ou diagonale)
- Placement récursif des reines avec une approche backtracking
- Affichage de la solution finale sur l'échiquier dans le terminal

## Prérequis
OCaml 

## Installation et exécution 
1. Cloner le dépôt:
``` powershell
git clone <url-du-repo>
cd <nom-du-repo>
```

2. Compiler et exécuter le fichier
```powershell 
ocamlc -o nreines Nreines.ml 
.\nreines
```

