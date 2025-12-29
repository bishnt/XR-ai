from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
from datetime import datetime
import torch

# Import our model and inference modules
from model import load_model
from inference import create_inference_engine

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Configuration
basedir = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(basedir, 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Global model and inference engine (loaded once at startup)
MODEL = None
INFERENCE_ENGINE = None


def initialize_model():
    """Initialize the model and inference engine at server startup"""
    global MODEL, INFERENCE_ENGINE
    
    print("=" * 60)
    print("Initializing X-Ray Classification Model...")
    print("=" * 60)
    
    try:
        # Check if CUDA is available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        if device == 'cuda':
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Version: {torch.version.cuda}")
        else:
            print("  Note: Running on CPU. Inference may be slower.")
        
        # Load model (with or without pretrained weights)
        # Assuming the user has placed the trained model in the backend directory
        MODEL = load_model(model_path='best_xray_model.pth', device=device)
        
        # Create inference engine
        INFERENCE_ENGINE = create_inference_engine(MODEL, device)
        
        print("✓ Model loaded successfully!")
        print(f"  Classes: {INFERENCE_ENGINE.class_names}")
        print("=" * 60)
        
        return True
    
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("  The /predict endpoint will not be available.")
        print("=" * 60)
        return False


def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to receive image and return X-ray classification prediction
    
    Expects: multipart/form-data with 'image' file
    Returns: JSON with prediction results
    """
    # Check if model is loaded
    if MODEL is None or INFERENCE_ENGINE is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded',
            'message': 'The classification model failed to load. Please check server logs.'
        }), 503
    
    try:
        # Check if the request contains a file
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image file provided',
                'message': 'Please include an image in the request'
            }), 400
        
        file = request.files['image']
        
        # Check if a file was actually selected
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected',
                'message': 'Please select a valid image file'
            }), 400
        
        # Validate file type
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': 'Invalid file type',
                'message': f'Allowed file types: {", ".join(ALLOWED_EXTENSIONS)}',
                'filename': file.filename
            }), 400
        
        # Secure the filename and save the file
        filename = secure_filename(file.filename)
        
        # Add timestamp to filename to avoid collisions
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        name, ext = os.path.splitext(filename)
        unique_filename = f"{name}_{timestamp}{ext}"
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Run inference
        prediction_result = INFERENCE_ENGINE.predict(filepath)
        
        # Get file information
        file_size = os.path.getsize(filepath)
        
        # Return success response with prediction
        return jsonify({
            'success': True,
            'message': 'Image analyzed successfully',
            'prediction': {
                'class': prediction_result['class'],
                'confidence': prediction_result['confidence'],
                'probabilities': prediction_result['probabilities'],
                'heatmap': prediction_result.get('heatmap')
            },
            'metadata': {
                'filename': unique_filename,
                'original_filename': file.filename,
                'filepath': filepath,
                'file_size': file_size,
                'file_size_mb': round(file_size / (1024 * 1024), 2),
                'processing_time_ms': prediction_result['processing_time_ms'],
                'timestamp': timestamp
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': 'Prediction error',
            'message': str(e)
        }), 500


@app.route('/receive', methods=['POST'])
def receive_image():
    """
    Endpoint to receive and process images from the frontend
    Expects a multipart/form-data POST request with an image file
    """
    try:
        # Check if the request contains a file
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image file provided',
                'message': 'Please include an image in the request'
            }), 400
        
        file = request.files['image']
        
        # Check if a file was actually selected
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected',
                'message': 'Please select a valid image file'
            }), 400
        
        # Validate file type
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': 'Invalid file type',
                'message': f'Allowed file types: {", ".join(ALLOWED_EXTENSIONS)}',
                'filename': file.filename
            }), 400
        
        # Secure the filename and save the file
        filename = secure_filename(file.filename)
        
        # Add timestamp to filename to avoid collisions
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        name, ext = os.path.splitext(filename)
        unique_filename = f"{name}_{timestamp}{ext}"
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Get file information
        file_size = os.path.getsize(filepath)
        
        # Return success response with file details
        return jsonify({
            'success': True,
            'message': 'Image received successfully',
            'data': {
                'filename': unique_filename,
                'original_filename': file.filename,
                'filepath': filepath,
                'file_size': file_size,
                'file_size_mb': round(file_size / (1024 * 1024), 2),
                'timestamp': timestamp
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': 'Server error',
            'message': str(e)
        }), 500


@app.route('/', methods=['GET'])
def index():
    """Root route to verify server is running"""
    return jsonify({
        'status': 'online',
        'service': 'XRAI Backend',
        'endpoints': {
            'predict': '/predict (POST)',
            'health': '/health (GET)'
        }
    }), 200


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    model_status = 'loaded' if MODEL is not None else 'not loaded'
    
    return jsonify({
        'status': 'healthy',
        'message': 'Flask server is running',
        'model_status': model_status,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'upload_folder': app.config['UPLOAD_FOLDER'],
        'max_file_size_mb': app.config['MAX_CONTENT_LENGTH'] / (1024 * 1024)
    }), 200


@app.errorhandler(413)
def file_too_large(e):
    """Handle file size exceeded error"""
    return jsonify({
        'success': False,
        'error': 'File too large',
        'message': f'Maximum file size is {MAX_FILE_SIZE / (1024 * 1024)}MB'
    }), 413


# Initialize model during import (important for Gunicorn)
initialize_model()

if __name__ == '__main__':
    print("=" * 60)
    print("Flask Server Starting...")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Max file size: {MAX_FILE_SIZE / (1024 * 1024)}MB")
    print(f"Allowed extensions: {', '.join(ALLOWED_EXTENSIONS)}")
    print("=" * 60)
    
    print("\nStarting Flask development server...")
    print("Available endpoints:")
    print("  - POST /predict    (X-ray classification)")
    print("  - POST /receive    (Upload only)")
    print("  - GET  /health     (Health check)")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)

