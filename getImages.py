import os
import requests
import base64
from bs4 import BeautifulSoup

def extract_and_save_images(html_file_path, output_folder):
    # Überprüfen, ob der Ausgabeordner existiert, und wenn nicht, erstellen
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # HTML-Datei öffnen und einlesen
    with open(html_file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # HTML-Inhalt mit Beautiful Soup analysieren
    soup = BeautifulSoup(content, 'html.parser')

    # Alle img-Tags finden
    images = soup.find_all('img')

    # Bilder extrahieren und speichern
    for i, img in enumerate(images):
        # Bild-URL erhalten
        src = img['src']

        # Annahme: Bilder sind in Base64 oder direkte URLs
        if src.startswith('data:image'):
            # Extrahieren des Bildformats und der Base64-Daten
            format, imgstr = src.split(';base64,')
            ext = format.split('/')[-1]

            # Bild aus Base64-Daten erstellen
            img_data = base64.b64decode(imgstr)
        else:
            # Bild von URL herunterladen
            response = requests.get(src)
            if response.status_code != 200:
                continue
            img_data = response.content
            ext = os.path.splitext(src)[1]

        # Bild speichern
        filename = f'image_{i}.{ext}'
        with open(os.path.join(output_folder, filename), 'wb') as f:
            f.write(img_data)
        print(f'Bild {i} gespeichert als {filename}')

# Pfad zur HTML-Datei und Ausgabeordner
html_file_path = 'final.html'  # Pfad zur HTML-Datei anpassen
output_folder = 'images_out/'  # Pfad zum Ausgabeordner anpassen

extract_and_save_images(html_file_path, output_folder)