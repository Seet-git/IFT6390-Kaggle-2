# IFT6390-Kaggle-2

Source: https://www.kaggle.com/competitions/ift3395-ift6390-identification-maladies-retine - 7e - Score 0.81488


# Détection de maladies rétiniennes - OCT

## Table des matières

1. [Installation des dépendances](#installation-des-dépendances)
2. [Configuration de Optuna avec MySQL et Ngrok](#configuration-de-optuna-avec-mysql-et-ngrok)
3. [Utilisation de wandb](#utilisation-de-wandb)
4. [Main et Configuration](#main-et-configuration)
5. [Fonctionnalités de Suivi des Logs et Résultats](#fonctionnalités-de-suivi-des-logs-et-résultats)
6. [Utilisation](#utilisation)

## Installation des Dépendances

Pour commencer, assurez-vous d'installer toutes les dépendances nécessaires à ce projet. Utilisez la commande suivante :

```bash
pip install -r requirements.txt
```

Note : Il est probable que vous devriez exécuter cette commande également : 

```bash
apt-get install -y libgl1-mesa-glx
```

Les dépendances incluent des bibliothèques telles que `numpy`, `torch`, `wandb`, `optuna`, etc.

Ensuite, téléchargez les données ici: https://www.kaggle.com/competitions/ift3395-ift6390-identification-maladies-retine/data puis déposez le fichier `.zip` dans le dossier ```./data```

## Configuration de Optuna avec MySQL et Ngrok

### 1. Configuration de la Base de Données MySQL

Pour optimiser vos modèles avec `optuna` et stocker les résultats dans une base de données, configurez MySQL comme suit :
1. **Installer MySQL:** Commencez par installer MySQL

   - Télécharger sur Windows / Mac / Linux: https://dev.mysql.com/downloads/mysql/

   - Terminal Linux:
      ```bash
      sudo apt-get update
      sudo apt-get install mysql-server
      ```
     
   Une fois téléchargé, vous devriez pouvoir utiliser cette commande : 
   ```bash
   sudo mysql -u root -p
   ```

2. **Créer une base de données MySQL :**
   Connectez-vous à MySQL et exécutez les commandes suivantes :
   
   ```sql
   CREATE DATABASE optuna_db;
   CREATE USER 'optuna_user'@'localhost' IDENTIFIED BY 'your_password';
   GRANT ALL PRIVILEGES ON optuna_db.* TO 'optuna_user'@'localhost';
   FLUSH PRIVILEGES;
   ```

3. **Autorisation et Accès :**
   - Assurez-vous que `optuna_user` a les autorisations nécessaires pour créer, lire, écrire et supprimer les entrées dans `optuna_db`.
   - Configurez les accès réseau de votre base de données si nécessaire.


Si tout est correct, vous devriez voir la base de données, avec la commande suivante :
   ```sql
    SHOW GRANTS FOR 'optuna_user'@'localhost';
    SHOW databases;
   ```

### 2. Utilisation de Ngrok pour une Connexion Externe (OPTIONNEL)

Pour rendre la base de données accessible à distance (utile si vous ne pouvez pas accéder à `localhost`), vous pouvez utiliser `ngrok` :

1. **Installer Ngrok et Ouvrir une Connexion avec Ngrok :**
   ```bash
   ngrok tcp 3306
   ```
   Remplacez `3306` par le port utilisé par votre serveur MySQL si différent.

## Utilisation de wandb

Wandb est utilisé pour suivre les expériences. Avant d'exécuter un script qui utilise `wandb`, connectez-vous en utilisant la commande suivante :

```bash
wandb login
```

Si vous décidez d'activer le suivi avec `wandb` dans vos scripts, assurez-vous que l'authentification est configurée correctement.

## Main et Configuration

Les programmes principaux (fichiers `baseline.py` et `start.py`) nécessitent une configuration pour les identifiants MySQL, le modèle, d'autres options. 
Pour configurer ces fichiers :

1. **Modifier le Fichier `config.py` :**
   - Entrer vos identifiants MySQL dans la section `LOGIN MYSQL`
   - Ngrok **uniquement**: Remplacer `localhost` par l'URL générée par Ngrok   

2. **Personnalisation :**
   - Vous pouvez modifier la valeur des paramètres selon vos besoins spécifiques.

## Fonctionnalités de Suivi des Logs et Résultats (OPTIONNEL)

Le projet dispose de plusieurs fonctionnalités de suivi :

### Optuna

Optuna dispose d'un dashboard pour visualiser les résultats, pour y accéder, il vous suffit d'exécuter la commande suivante:
```bash
pip install optuna-dashboard
optuna-dashboard "mysql+pymysql://optuna_user:your_password@localhost:3306/optuna_db"
```

## Utilisation

### Script principal

1. Exécutez le script principal : `python start.py`

La fonction `main()` effectue les tâches suivantes :

- Chargement des données d'entraînement et de test avec `extract_data.py`.
- Recherche bayésienne.
- Entraînement du modèle.
- Évaluation des performances du modèle sur les données de test.

### Baseline

1) Extraire les données dans le dossier `./data`

2) Exécutez la baseline : `python baseline.py`

La fonction `main()` dans baseline.py effectue :

 - Chargement des données à l'aide de NumPy.

 - Recherche de grille pour l'optimisation des hyper-paramètres avec Grid Search.

 - Entraînement du modèle.

 - Évaluation du modèle sur le jeu de test.