from flask import Flask, request, jsonify
from flask_cors import CORS
from deepface import DeepFace
import cv2
import numpy as np
import base64
import io

app = Flask(__name__)
CORS(app)

@app.route('/verify', methods=['POST'])
def verify():
    data = request.json
    img_data = base64.b64decode(data['image'])
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (300, 300))  # Resize agar lebih cepat

    # Deteksi wajah (menggunakan DeepFace)
    detections = DeepFace.extract_faces(
        img_path=img,
        detector_backend="opencv",
        enforce_detection=False,
        align=False
    )

    # Tambahkan frame pada wajah yang terdeteksi
    for det in detections:
        x, y, w, h = det['facial_area'].values()
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Encode gambar dengan frame ke base64
    _, buffer = cv2.imencode('.jpg', img)
    img_with_frame = base64.b64encode(buffer).decode('utf-8')

    # Proses identifikasi seperti biasa
    result = DeepFace.find(
        img_path=img,
        db_path="faces/",
        detector_backend="opencv",
        model_name="Facenet",
        enforce_detection=False
    )
    if len(result) > 0:
        identity = result.iloc[0]['identity']
        return jsonify({"status": "success", "identity": identity, "framed_image": img_with_frame})
    else:
        return jsonify({"status": "not_found", "framed_image": img_with_frame})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)