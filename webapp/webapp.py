import os
import sys

# append parent
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append( parent_dir)
sys.path.append( os.path.join(parent_dir, 'src') )

from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
import torch
from models.siamese_network import SiameseNetwork
from helpers import allowed_file, get_uploaded_files, preprocess_image
import constants
from datetime import datetime

app = Flask(__name__, template_folder='templates')

# Directories
UPLOAD_FOLDER = 'data_source/images/uploads'
MODEL_PATH = '../saved_models/trained_model.pth'

# Dynamically resolve the training dataset folder
base_path = os.path.dirname(os.path.abspath(__file__))
training_dataset_folder = os.path.join(parent_dir, 'data_source_min', 'images', 'fish_02')
print(f"Training dataset folder: {training_dataset_folder}")


# Allowed extensions for uploads
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = SiameseNetwork()
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        print("Model successfully loaded.")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
else:
    print(f"Error: Model file not found at {MODEL_PATH}. Please train the model first.")

@app.route('/')
def index():
    uploaded_files = get_uploaded_files(app.config['UPLOAD_FOLDER'])
    return render_template('home.html', uploaded_files=uploaded_files)


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file selected"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No file selected"}), 400

    if file and allowed_file(file.filename, app.config['ALLOWED_EXTENSIONS']):
        try:
            # Save the uploaded file
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            file.save(save_path)

            if not model:
                return jsonify({"status": "error", "message": "Model not loaded"}), 500

            # Process the uploaded image
            with torch.no_grad():
                uploaded_tensor = preprocess_image(save_path).unsqueeze(0)

                # Comparison with uploaded images
                results = []
                for ref_filename in os.listdir(app.config['UPLOAD_FOLDER']):
                    if ref_filename != filename:  # Avoid comparing with itself
                        ref_path = os.path.join(app.config['UPLOAD_FOLDER'], ref_filename)
                        ref_tensor = preprocess_image(ref_path).unsqueeze(0)
                        distance = model(uploaded_tensor, ref_tensor)
                        results.append({
                            "filename": ref_filename,
                            "distance": float(distance[0].item())
                        })

                # Comparison with training dataset
                for train_filename in os.listdir(training_dataset_folder):
                    train_path = os.path.join(training_dataset_folder, train_filename)
                    train_tensor = preprocess_image(train_path).unsqueeze(0)
                    distance = model(uploaded_tensor, train_tensor)
                    results.append({
                        "filename": train_filename,
                        "distance": float(distance[0].item())
                    })

                # Sort results by distance (most similar first)
                results.sort(key=lambda x: x["distance"])

                return jsonify({
                    "status": "success",
                    "results": results
                })

        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

    return jsonify({"status": "error", "message": "Invalid file type"}), 400


@app.route('/download/<filename>')
def download_file(filename):
    upload_dir = os.path.join(os.getcwd(), 'src', 'api', 'data_source', 'images', 'fish_01')
    return send_from_directory(upload_dir, filename, as_attachment=False)

@app.template_filter('datetime')
def format_datetime(value):
    return datetime.fromtimestamp(value).strftime('%Y-%m-%d %H:%M:%S')

if __name__ == '__main__':
    app.run(debug=True)
