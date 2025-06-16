from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    send_from_directory,
    flash,
)
from werkzeug.utils import secure_filename
import os
import numpy as np
import re
import base64
import cv2
from tensorflow.keras.models import load_model as keras_load_model

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "./static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}


# ===== Load Trained Model =====
def load_model():
    model_path = "./model/mnistModel.keras"
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file not found!")
    model = keras_load_model(model_path)
    print("Model loaded successfully!")
    return model


model = load_model()


# ===== Utilities =====
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.bitwise_not(img)
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(img)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        img = img[y : y + h, x : x + w]
    img = cv2.resize(img, (20, 20), interpolation=cv2.INTER_AREA)
    img = cv2.copyMakeBorder(img, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0)
    img = img.astype("float32") / 255.0
    img = img.reshape(1, 28, 28, 1)
    return img


# ===== Routes =====
@app.route("/")
def index():
    return render_template("index.html")


# @app.route("/upload", methods=["POST"])
# def upload():
#     if "file" not in request.files:
#         flash("No file part")
#         return redirect(url_for("index"))
#     file = request.files["file"]
#     if file.filename == "":
#         flash("No selected file")
#         return redirect(url_for("index"))
#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
#         file.save(filepath)

#         img = preprocess_image(filepath)
#         prediction = model.predict(img)
#         digit = int(np.argmax(prediction))
#         confidence = float(np.max(prediction))

#         return f"<h1>Prediction: {digit} (Confidence: {confidence:.2f})</h1>"

#     return "File not allowed", 400


@app.route("/predict", methods=["POST"])
def predict_canvas():
    try:
        img_data = request.get_data()
        img_str = re.search(b"base64,(.*)", img_data).group(1)
        with open("digit.png", "wb") as output:
            output.write(base64.decodebytes(img_str))

        img = preprocess_image("digit.png")
        prediction = model.predict(img)
        digit = int(np.argmax(prediction))
        return str(digit)
    except Exception as e:
        return f"Error: {e}", 500


if __name__ == "__main__":
    app.run(debug=True)
