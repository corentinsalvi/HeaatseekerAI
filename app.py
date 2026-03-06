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
from flask import Flask, request, jsonify, send_from_directory, session, redirect
from functools import wraps
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
USERS_DIR    = "users_data"
IMG_SIZE     = (224, 224)
os.makedirs(USERS_DIR, exist_ok=True)

# Le modèle est chargé par PredictionLegit.py au moment de l'import

app = Flask(__name__, static_folder="static", template_folder=".")
CORS(app)
app.secret_key = 'heaatseeker-secret-2025'

import json
import hashlib

USERS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'users.json')

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    # Compte admin par défaut si le fichier n'existe pas encore
    default = {'admin': hash_password('heaat2025')}
    save_users(default)
    return default

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect('/login')
        return f(*args, **kwargs)
    return decorated

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

# ── CNN Prediction + GradCAM ──────────────────────────────────────────────────
def predict_images(image_paths: list[str], out_dir: str) -> dict:
    """Prédit et génère les GradCAM directement dans out_dir."""
    if not image_paths:
        return {"error": "No images to predict"}

    if not os.path.exists(MODEL_PATH):
        return {"error": f"Modèle introuvable : {MODEL_PATH}"}

    try:
        from tensorflow.keras.models import load_model as _load
        from GradCam import get_conv_layer_names, process_image as gradcam_process
        m = _load(MODEL_PATH)
        conv_names = get_conv_layer_names(m)
    except Exception:
        log.error(traceback.format_exc())
        return {"error": "Impossible de charger le modèle ou GradCam"}

    fake_count  = 0
    legit_count = 0
    confidences = []
    images_info = []   # [{original, gradcam, pred}]

    for img_path in image_paths:
        try:
            img = load_img(img_path, target_size=(224, 224))
            arr = img_to_array(img) / 255.0
            tensor = np.expand_dims(arr, axis=0)
            pred = float(m.predict(tensor, verbose=0)[0][0])
            confidences.append(pred * 100)

            if pred >= 0.5:
                fake_count += 1
            else:
                legit_count += 1

            # GradCAM — sauvegardé dans out_dir
            gradcam_path = None
            try:
                _, gradcam_path = gradcam_process(
                    img_path, m,
                    target_size=(224, 224),
                    conv_candidate_names=conv_names,
                    out_dir=out_dir,
                    alpha=0.4
                )
            except Exception as e:
                log.warning(f"GradCAM failed for {img_path}: {e}")

            images_info.append({
                "original": "/files/" + img_path.replace(os.sep, "/"),
                "gradcam":  "/files/" + gradcam_path.replace(os.sep, "/") if gradcam_path else None,
                "pred":     round(pred * 100, 1),
                "vote":     "FAKE" if pred >= 0.5 else "LEGIT"
            })

        except Exception:
            log.warning(f"Could not predict {img_path}")

    if not confidences:
        return {"error": "Prediction failed for all images"}

    avg_confidence = float(np.mean(confidences))

    if avg_confidence < 50:
        result = "INCONNU"
    else:
        result = "FAKE" if fake_count > legit_count else "LEGIT"

    return {
        "result":         result,
        "avg_confidence": round(avg_confidence, 2),
        "legit_count":    int(legit_count),
        "fake_count":     int(fake_count),
        "total":          int(fake_count + legit_count),
        "low_confidence": bool(avg_confidence < 50),
        "images_info":    images_info,
        "mock":           False,
    }

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
@login_required
def index():
    base = os.path.dirname(os.path.abspath(__file__))
    for folder in [base, os.path.join(base, "static")]:
        if os.path.exists(os.path.join(folder, "heaatseeker.html")):
            return send_from_directory(folder, "heaatseeker.html")
    return "Placez heaatseeker.html dans le meme dossier que app.py", 404

@app.route("/login")
def login_page():
    if session.get("logged_in"):
        return redirect("/")
    base = os.path.dirname(os.path.abspath(__file__))
    for folder in [base, os.path.join(base, "static")]:
        if os.path.exists(os.path.join(folder, "login.html")):
            return send_from_directory(folder, "login.html")
    return "Placez login.html dans le meme dossier que app.py", 404

@app.route("/register")
def register_page():
    if session.get("logged_in"):
        return redirect("/")
    base = os.path.dirname(os.path.abspath(__file__))
    for folder in [base, os.path.join(base, "static")]:
        if os.path.exists(os.path.join(folder, "register.html")):
            return send_from_directory(folder, "register.html")
    return "Placez register.html dans le meme dossier que app.py", 404

@app.route("/api/login", methods=["POST"])
def api_login():
    data = request.get_json(force=True)
    username = (data.get("username") or "").strip()
    password = (data.get("password") or "")
    users = load_users()
    if users.get(username) == hash_password(password):
        session["logged_in"] = True
        session["username"] = username
        # Créer le dossier utilisateur s'il n'existe pas
        user_dir = os.path.join(USERS_DIR, username)
        os.makedirs(user_dir, exist_ok=True)
        return jsonify({"ok": True})
    return jsonify({"ok": False, "error": "Identifiant ou mot de passe incorrect."}), 401

@app.route("/api/register", methods=["POST"])
def api_register():
    data = request.get_json(force=True)
    username = (data.get("username") or "").strip()
    password = (data.get("password") or "")
    if not username or not password:
        return jsonify({"ok": False, "error": "Champs manquants."}), 400
    if len(password) < 6:
        return jsonify({"ok": False, "error": "Mot de passe trop court (6 caractères min)."}), 400
    users = load_users()
    if username in users:
        return jsonify({"ok": False, "error": "Cet identifiant est déjà pris."}), 409
    users[username] = hash_password(password)
    save_users(users)
    session["logged_in"] = True
    session["username"] = username
    # Créer le dossier utilisateur
    user_dir = os.path.join(USERS_DIR, username)
    os.makedirs(user_dir, exist_ok=True)
    return jsonify({"ok": True})

@app.route("/api/logout", methods=["POST"])
def api_logout():
    session.clear()
    return jsonify({"ok": True})

@app.route("/api/authenticate-upload", methods=["POST"])
@login_required
def authenticate_upload():
    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "Aucune image reçue"}), 400

    username = session.get("username", "anonymous")
    # Sous-dossier par analyse : users_data/<username>/<uuid>/
    analysis_id = str(uuid.uuid4())
    user_dir = os.path.join(USERS_DIR, username)
    out_dir  = os.path.join(user_dir, analysis_id)
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

    result = predict_images(paths, out_dir)
    result["analysis_id"] = analysis_id
    result["username"] = username
    return jsonify(result)

@app.route("/api/authenticate", methods=["POST"])
@login_required
def authenticate():
    data = request.get_json(force=True)
    url  = (data.get("url") or "").strip()
    if not url:
        return jsonify({"error": "URL manquante"}), 400

    site = detect_site(url)
    if site == "unknown":
        return jsonify({"error": "Site non supporté (Vinted/Grailed uniquement)"}), 400

    username    = session.get("username", "anonymous")
    analysis_id = str(uuid.uuid4())
    user_dir    = os.path.join(USERS_DIR, username)
    out_dir     = os.path.join(user_dir, analysis_id)
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
        result = predict_images(paths, out_dir)
        result["site"]        = site
        result["analysis_id"] = analysis_id
        result["username"]    = username
        return jsonify(result)

    except Exception:
        log.error(traceback.format_exc())
        shutil.rmtree(out_dir, ignore_errors=True)
        return jsonify({"error": "Erreur interne lors du scraping"}), 500

@app.route("/files/<path:filepath>")
@login_required
def serve_file(filepath):
    """Sert les fichiers depuis users_data (images + gradcam)."""
    base = os.path.dirname(os.path.abspath(__file__))
    full = os.path.join(base, filepath)
    if not os.path.exists(full):
        return "File not found", 404
    return send_from_directory(os.path.dirname(full), os.path.basename(full))

if __name__ == "__main__":
    app.run(debug=True, port=5000)
