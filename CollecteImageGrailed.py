import os
import requests
import logging
import traceback
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time
import sys
 
# Configuration simple du logging (niveau DEBUG si DEBUG=1 dans l'environnement)
log_level = logging.DEBUG if os.environ.get("DEBUG") == "1" else logging.INFO
logging.basicConfig(level=log_level, format='%(asctime)s %(levelname)s: %(message)s')

# Fonction pour télécharger les images en format jpg dans le dossier spécifié
def download_image(image_url, folder_path):
    try:
        # Extraire le nom de l'image depuis l'URL
        image_name = image_url.split("/")[-1].split("?")[0]
        
        # S'assurer que l'image est en .jpg (si l'extension n'est pas déjà .jpg, la modifier)
        if not image_name.lower().endswith(".jpg"):
            # retirer l'extension actuelle et forcer .jpg
            image_base = os.path.splitext(image_name)[0]
            image_name = image_base + ".jpg"
        
        image_path = os.path.join(folder_path, image_name)
        
        # Télécharger l'image et l'enregistrer
        logging.debug(f"Téléchargement: {image_url} -> {image_path}")
        response = requests.get(image_url, timeout=15)
        if response.status_code == 200:
            with open(image_path, 'wb') as file:
                file.write(response.content)
            logging.info(f"Image téléchargée: {image_name}")
            return True
        else:
            logging.warning(f"Erreur de téléchargement pour {image_url} - status {response.status_code}")
            return False
    except Exception:
        logging.error(f"Erreur avec l'image {image_url}:\n" + traceback.format_exc())
        return False

# Fonction pour extraire les images depuis une page Grailed
def extract_images_from_grailed(page_url):
    logging.info(f"Démarrage extraction depuis: {page_url}")

    driver = None
    folder_path = os.path.join(os.getcwd(), "Image AJ1 Travis Scott Grailed")
    try:
        # Initialiser Selenium WebDriver avec Chrome
        logging.info("Initialisation du WebDriver Chrome (webdriver_manager) ...")
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

        # Ouvrir la page principale d'articles
        logging.info(f"Ouverture de la page principale: {page_url}")
        driver.get(page_url)
        time.sleep(5)  # Attendre que la page charge (améliorable)

        # Récupérer le code HTML de la page
        soup = BeautifulSoup(driver.page_source, 'html.parser')

        # Trouver tous les liens d'articles (href commençant par /listings/)
        links = soup.select('a[href^="/listings/"]')
        # Filtrer les doublons (même href) et garder l'ordre
        seen_hrefs = set()
        articles = []
        for a in links:
            href = a.get('href')
            if not href:
                continue
            if href in seen_hrefs:
                continue
            seen_hrefs.add(href)
            articles.append(a)
        logging.info(f"Articles (liens) trouvés sur la page principale: {len(articles)}")

        # Debug: si aucun article trouvé, sauvegarder la page pour inspection
        if len(articles) == 0:
            try:
                debug_path = os.path.join(os.getcwd(), 'debug_page_main.html')
                with open(debug_path, 'w', encoding='utf-8') as f:
                    f.write(driver.page_source)
                logging.warning(f"Aucun article trouvé — page principale sauvegardée pour debug: {debug_path}")
            except Exception:
                logging.error("Échec lors de la sauvegarde de la page principale pour debug:\n" + traceback.format_exc())

        # Créer un dossier pour les images
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            logging.info(f"Dossier créé: {folder_path}")
        else:
            logging.info(f"Dossier existant: {folder_path}")

        # Parcourir chaque article (élément <a>)
        # Nous allons d'abord télécharger l'image de couverture si présente
        downloaded_urls = set()
        for idx, article in enumerate(articles, start=1):
            try:
                # 'article' est un <a> du flux; son href contient le lien vers la page de listing
                href = article.get('href')
                if not href:
                    logging.debug(f"Article #{idx} sans href, skip")
                    continue
                article_url = 'https://www.grailed.com' + href
                logging.info(f"[{idx}/{len(articles)}] Article href: {href}")

                # Télécharger l'image de couverture présente dans le <a> (si elle existe)
                img_tag = article.find('img')
                cover_downloaded = 0
                if img_tag:
                    img_src = img_tag.get('src') or ''
                    # si src est vide, essayer srcset
                    if (not img_src) and img_tag.get('srcset'):
                        # prendre la première URL du srcset
                        img_src = img_tag.get('srcset').split()[0]
                    if img_src:
                        logging.debug(f"Article {idx} - image de couverture trouvée: {img_src}")
                        if img_src not in downloaded_urls:
                            if download_image(img_src, folder_path):
                                downloaded_urls.add(img_src)
                                cover_downloaded = 1
                logging.info(f"Article {idx} - couverture téléchargée: {cover_downloaded}")

                # Ensuite, ouvrir la page d'article pour récupérer toutes les images medias
                try:
                    driver.get(article_url)
                except Exception:
                    logging.error(f"Erreur lors de l'ouverture de la page {article_url}:\n" + traceback.format_exc())
                    continue

                time.sleep(3)
                article_soup = BeautifulSoup(driver.page_source, 'html.parser')

                # Récupérer toutes les <img> de la page et télécharger celles qui semblent être des assets de listing
                page_imgs = article_soup.find_all('img')
                media_images = []
                for img in page_imgs:
                    src = img.get('src') or ''
                    if not src and img.get('srcset'):
                        src = img.get('srcset').split()[0]
                    if src and 'media-assets.grailed.com' in src:
                        media_images.append(src)

                logging.info(f"Article {idx} - images media détectées sur page: {len(media_images)}")
                downloaded = 0
                for im_idx, image_url in enumerate(media_images, start=1):
                    if image_url in downloaded_urls:
                        logging.debug(f"Article {idx} - image #{im_idx} déjà téléchargée, skip")
                        continue
                    logging.debug(f"Article {idx} - téléchargement image #{im_idx}: {image_url}")
                    if download_image(image_url, folder_path):
                        downloaded += 1
                        downloaded_urls.add(image_url)
                logging.info(f"Article {idx} - images téléchargées depuis page: {downloaded}/{len(media_images)}")
            except Exception:
                logging.error(f"Erreur lors du traitement de l'article #{idx}:\n" + traceback.format_exc())

    except Exception:
        logging.error("Erreur fatale lors de l'initialisation ou de l'exécution:\n" + traceback.format_exc())
    finally:
        # Fermer le navigateur après l'exécution
        if driver is not None:
            try:
                driver.quit()
                logging.info("WebDriver fermé proprement.")
            except Exception:
                logging.warning("Erreur lors de la fermeture du WebDriver:\n" + traceback.format_exc())

# URL de la page des articles Grailed que tu veux analyser
page_url = 'https://www.grailed.com/shop?query=travis%20scott%20jordan%201%20phantom'

# Appeler la fonction pour extraire et télécharger les images
extract_images_from_grailed(page_url)
