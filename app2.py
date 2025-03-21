import ssl
import certifi

ssl._create_default_https_context = ssl._create_unverified_context

from flask import Flask, request, jsonify
from flask_cors import CORS
import easyocr
import requests
import numpy as np
import cv2
from spellchecker import SpellChecker
from fuzzywuzzy import process
import os  # Add this


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

reader = easyocr.Reader(['en'])
spell = SpellChecker()

# Load medicine names
def load_medicine_list():
    try:
        with open("medicines.txt", "r") as file:
            return [line.strip() for line in file.readlines()]
    except FileNotFoundError:
        return []

medicine_list = load_medicine_list()

def correct_text(text):
    words = text.split()
    corrected_words = []
    for word in words:
        corrected_word = spell.correction(word) or word
        best_match, score = process.extractOne(corrected_word, medicine_list) if medicine_list else (corrected_word, 0)
        corrected_words.append(best_match if score > 80 else corrected_word)
    return " ".join(corrected_words)

def download_image(image_url):
    try:
        response = requests.get(image_url, stream=True)
        response.raise_for_status()
        img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        return None

@app.route("/ocr", methods=["POST"])
def ocr():
    data = request.json
    image_url = data.get("image_url")

    if not image_url:
        return jsonify({"error": "No image URL provided"}), 400

    img = download_image(image_url)
    if img is None:
        return jsonify({"error": "Failed to load image"}), 400

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    text_results = reader.readtext(img_gray, detail=0)

    if not text_results:
        return jsonify({"error": "No text found"}), 400

    extracted_text = " ".join(text_results)
    corrected_text = correct_text(extracted_text)

    return jsonify({"text": corrected_text})

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))  # Use PORT from Render
    app.run(host="0.0.0.0", port=port, debug=False)