from selenium import webdriver
from selenium.webdriver.common.by import By
import os
import requests
import time

def telecharger_image(url_image, chemin_fichier):
    """
    Télécharge une image depuis une URL et l'enregistre dans un fichier.
    """
    try:
        response = requests.get(url_image, stream=True)
        if response.status_code == 200:
            with open(chemin_fichier, "wb") as f:
                f.write(response.content)
            print(f"Image téléchargée : {chemin_fichier}")
        else:
            print(f"Erreur HTTP {response.status_code} pour {url_image}")
    except Exception as e:
        print(f"Erreur lors du téléchargement de l'image : {e}")

def extraire_images_article_selenium(driver, url_article, dossier_destination):
    """
    Récupère et télécharge toutes les images des balises <figure> dans un article.
    """
    driver.get(url_article)
    time.sleep(2)  # Attendre que la page se charge

    # Trouver toutes les balises <figure> contenant des images
    figures = driver.find_elements(By.TAG_NAME, "figure")
    if not os.path.exists(dossier_destination):
        os.makedirs(dossier_destination)

    for i, figure in enumerate(figures):
        img = figure.find_element(By.TAG_NAME, "img")
        img_url = img.get_attribute("src")
        if not img_url:
            continue

        try:
            chemin_fichier = os.path.join(dossier_destination, f"image_{i+1}.jpg")
            telecharger_image(img_url, chemin_fichier)
        except Exception as e:
            print(f"Erreur lors du téléchargement de {img_url}: {e}")

def extraire_toutes_les_images_selenium(url, dossier_destination="images"):
    """
    Récupère toutes les images des articles depuis une page Vinted avec Selenium.
    """
    driver = webdriver.Chrome()

    try:
        driver.get(url)
        time.sleep(5)  # Attendre que la page et JavaScript se chargent

        # Récupérer les liens des articles
        liens_articles = []
        liens = driver.find_elements(By.TAG_NAME, "a")
        for lien in liens:
            href = lien.get_attribute("href")
            if href and "https://www.vinted.fr/items/" in href:
                liens_articles.append(href)

        liens_articles = list(set(liens_articles))  # Supprimer les doublons
        print(f"{len(liens_articles)} articles trouvés.")

        if not liens_articles:
            print("Aucun lien d'article trouvé.")
            return

        for i, lien_article in enumerate(liens_articles):
            print(f"Traitement de l'article {i+1}/{len(liens_articles)} : {lien_article}")
            dossier_article = os.path.join(dossier_destination, f"article_{i+1}")
            extraire_images_article_selenium(driver, lien_article, dossier_article)

    finally:
        driver.quit()

# Exemple d'utilisation
url_page_principale = input("Entrez l'URL de la page Vinted : ")
extraire_toutes_les_images_selenium(url_page_principale)
