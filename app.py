"""
Heatseeker AI - Backend Flask
Scrape images from Vinted/Grailed listings and run CNN authenticity prediction.
"""

import os
import time
import uuid
import shutil
import logging
import traceback
import threading

import numpy as np
import requests
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO

# ── Tensorflow / Keras ─────────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_PATH   = os.environ.get("MODEL_PATH", "reseauncnn_best.keras")
UPLOAD_DIR   = "static/sessions"
IMG_SIZE     = (224, 224)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Le modèle est chargé par PredictionLegit.py au moment de l'import

app = Flask(__name__, static_folder="static", template_folder=".")
CORS(app)

# ── Helper: detect site ────────────────────────────────────────────────────────
def detect_site(url: str) -> str:
    if "vinted.fr" in url or "vinted.com" in url:
        return "vinted"
    if "grailed.com" in url:
        return "grailed"
    return "unknown"

# ── Helper: build Chrome driver ────────────────────────────────────────────────
def build_driver():
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--window-size=1280,900")
    opts.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
    try:
        from webdriver_manager.chrome import ChromeDriverManager
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opts)
    except Exception:
        driver = webdriver.Chrome(options=opts)
    return driver

# ── Scraper: Vinted ────────────────────────────────────────────────────────────
def scrape_vinted(url: str, out_dir: str) -> list[str]:
    driver = build_driver()
    paths = []
    try:
        driver.get(url)
        time.sleep(4)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        figures = soup.find_all("figure")
        for i, fig in enumerate(figures):
            img = fig.find("img")
            if not img:
                continue
            src = img.get("src") or (img.get("srcset", "").split()[0] if img.get("srcset") else "")
            if not src:
                continue
            p = _download_image(src, out_dir, f"img_{i:03d}.jpg")
            if p:
                paths.append(p)
    finally:
        driver.quit()
    return paths

# ── Scraper: Grailed ───────────────────────────────────────────────────────────
def scrape_grailed(url: str, out_dir: str) -> list[str]:
    driver = build_driver()
    paths = []
    try:
        driver.get(url)
        time.sleep(5)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        imgs = soup.find_all("img")
        i = 0
        for img in imgs:
            src = img.get("src") or (img.get("srcset", "").split()[0] if img.get("srcset") else "")
            if src and "media-assets.grailed.com" in src:
                p = _download_image(src, out_dir, f"img_{i:03d}.jpg")
                if p:
                    paths.append(p)
                    i += 1
    finally:
        driver.quit()
    return paths

# ── Image downloader ───────────────────────────────────────────────────────────
def _download_image(url: str, folder: str, filename: str) -> str | None:
    try:
        resp = requests.get(url, timeout=15,
                            headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code != 200:
            return None
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        path = os.path.join(folder, filename)
        img.save(path, "JPEG")
        return path
    except Exception:
        return None

# ── CNN Prediction via PredictionLegit.py ─────────────────────────────────────
def predict_images(image_paths: list[str]) -> dict:
    if not image_paths:
        return {"error": "No images to predict"}

    # Sauvegarder les images dans un dossier temporaire
    # puis appeler predict_article de PredictionLegit
    import tempfile, shutil
    try:
        from PredictionLegit import predict_article
    except ImportError as e:
        return {"error": f"Impossible d'importer PredictionLegit.py : {e}"}

    # Créer un dossier temporaire contenant les images
    tmp_dir = tempfile.mkdtemp()
    try:
        for p in image_paths:
            shutil.copy(p, tmp_dir)

        result, fake_count, legit_count, avg_confidence = predict_article(tmp_dir)

        return {
            "result": result,
            "avg_confidence": round(float(avg_confidence), 2),
            "legit_count": int(legit_count),
            "fake_count":  int(fake_count),
            "total": int(fake_count + legit_count),
            "low_confidence": bool(avg_confidence < 50),
            "mock": False,
        }
    except Exception:
        log.error(traceback.format_exc())
        return {"error": "Erreur lors de la prédiction"}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    import os
    base = os.path.dirname(os.path.abspath(__file__))
    for folder in [base, os.path.join(base, "static")]:
        if os.path.exists(os.path.join(folder, "heaatseeker.html")):
            return send_from_directory(folder, "heaatseeker.html")
    return "Placez heaatseeker.html dans le même dossier que app.py", 404

@app.route("/api/authenticate-upload", methods=["POST"])
def authenticate_upload():
    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "Aucune image reçue"}), 400

    session_id = str(uuid.uuid4())
    out_dir = os.path.join(UPLOAD_DIR, session_id)
    os.makedirs(out_dir, exist_ok=True)

    paths = []
    for i, f in enumerate(files):
        if not f.filename:
            continue
        try:
            img = Image.open(f.stream).convert("RGB")
            path = os.path.join(out_dir, f"img_{i:03d}.jpg")
            img.save(path, "JPEG")
            paths.append(path)
        except Exception:
            log.warning(f"Could not process uploaded file {f.filename}")

    if not paths:
        return jsonify({"error": "Aucune image valide dans l'envoi"}), 422

    result = predict_images(paths)
    result["images"] = [
        f"/static/sessions/{session_id}/{os.path.basename(p)}" for p in paths
    ]
    return jsonify(result)

@app.route("/api/authenticate", methods=["POST"])
def authenticate():
    data = request.get_json(force=True)
    url  = (data.get("url") or "").strip()
    if not url:
        return jsonify({"error": "URL manquante"}), 400

    site = detect_site(url)
    if site == "unknown":
        return jsonify({"error": "Site non supporté (Vinted/Grailed uniquement)"}), 400

    session_id = str(uuid.uuid4())
    out_dir = os.path.join(UPLOAD_DIR, session_id)
    os.makedirs(out_dir, exist_ok=True)

    try:
        log.info(f"Scraping {site}: {url}")
        if site == "vinted":
            paths = scrape_vinted(url, out_dir)
        else:
            paths = scrape_grailed(url, out_dir)

        if not paths:
            return jsonify({"error": "Aucune image trouvée sur cette annonce"}), 422

        log.info(f"Scraped {len(paths)} images → predicting…")
        result = predict_images(paths)
        result["site"] = site
        result["session_id"] = session_id
        result["images"] = [
            f"/static/sessions/{session_id}/{os.path.basename(p)}" for p in paths
        ]
        return jsonify(result)

    except Exception:
        log.error(traceback.format_exc())
        shutil.rmtree(out_dir, ignore_errors=True)
        return jsonify({"error": "Erreur interne lors du scraping"}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
