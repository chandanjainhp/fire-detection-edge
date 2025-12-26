from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import json
from fire_detection_backend import FireDetectionModel
import logging

app = Flask(__name__)
CORS(app)

# Configure logging for debugging on remote devices
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'gif'}
MODEL_PATH = 'fire_detection_model.h5'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max (reduced from 16MB)

# Initialize model (prefer quantized for edge devices)
detector = FireDetectionModel(MODEL_PATH)
model_loaded = detector.load_trained_model(prefer_quantized=True)

# Auto train if model not loaded and data directory exists
if not model_loaded and os.path.exists('./data'):
    logger.info("No model found. Training model automatically...")
    detector.train('./data', epochs=15, batch_size=16)
    model_loaded = True
    logger.info("Model trained and saved.")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status', methods=['GET'])
def status():
    """Get system status and model information"""
    model_info = detector.get_model_info()
    return jsonify({
        'status': 'online',
        'model_loaded': model_loaded,
        'model_info': model_info,
        'device': 'edge_optimized',
        'max_file_size_mb': 10
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict fire from uploaded image"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: png, jpg, jpeg, bmp, gif'}), 400

    try:
        if not model_loaded:
            return jsonify({'error': 'Model not loaded'}), 500

        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Predict
        logger.info(f"Processing image: {filename}")
        result = detector.predict_image(filepath)

        # Send alert if fire detected
        send_alert = request.form.get('send_alert', 'false').lower() == 'true'
        alert_email = request.form.get('alert_email', '')

        if result['is_fire'] and send_alert and alert_email:
            alert_sent = detector.send_email_alert(alert_email, result)
            result['alert_sent'] = alert_sent

        # Clean up uploaded file
        os.remove(filepath)

        logger.info(f"Prediction: {result['class']} ({result['confidence']:.2%})")
        return jsonify(result)

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        # Clean up on error
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict_stream', methods=['POST'])
def predict_stream():
    """Predict fire from image bytes (for camera streams)"""
    try:
        if not model_loaded:
            return jsonify({'error': 'Model not loaded'}), 500

        image_bytes = request.get_data()

        if len(image_bytes) == 0:
            return jsonify({'error': 'No image data received'}), 400

        result = detector.predict_from_bytes(image_bytes)
        return jsonify(result)

    except Exception as e:
        logger.error(f"Stream prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/train', methods=['POST'])
def train():
    """Train new model (use with caution on edge devices - may take time)"""
    try:
        data = request.get_json()
        data_directory = data.get('data_directory', './data')
        epochs = data.get('epochs', 15)
        batch_size = data.get('batch_size', 16)

        if not os.path.exists(data_directory):
            return jsonify({
                'error': f'Data directory {data_directory} does not exist. '
                        'Organize your training images into subfolders (fire/no_fire).'
            }), 400

        logger.info(f"Starting training: {epochs} epochs, batch size {batch_size}")

        global detector, model_loaded
        detector = FireDetectionModel(MODEL_PATH)
        history = detector.train(data_directory, epochs, batch_size)
        model_loaded = True

        return jsonify({
            'success': True,
            'message': 'Model trained successfully',
            'final_accuracy': float(history.history['accuracy'][-1]),
            'final_val_accuracy': float(history.history['val_accuracy'][-1]),
            'model_info': detector.get_model_info()
        })

    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/model_info', methods=['GET'])
def model_info():
    """Get detailed model information"""
    return jsonify(detector.get_model_info())

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint for monitoring"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'timestamp': detector.predict_from_bytes if model_loaded else None
    })

if __name__ == '__main__':
    # For edge devices, run on all interfaces
    # Set threaded=True for handling multiple requests
    app.run(
        debug=False,  # Set to False in production
        host='0.0.0.0',
        port=5000,
        threaded=True
    )
