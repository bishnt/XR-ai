# X-Ray Classification Backend

## Setup Instructions

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

**Note**: PyTorch installation may take several minutes and requires ~1-2GB download.

For GPU support (optional), install the CUDA-enabled version:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 2. Run the Server

```bash
python app.py
```

The server will start on `http://localhost:5000`

## API Endpoints

### POST /predict
Classifies an X-ray image

**Request**:
- Content-Type: multipart/form-data
- Body: image file (png, jpg, jpeg, gif, bmp, tiff, webp)

**Response**:
```json
{
  "success": true,
  "message": "Image analyzed successfully",
  "prediction": {
    "class": "NORMAL",
    "confidence": 0.94,
    "probabilities": {
      "NORMAL": 0.94,
      "PNEUMONIA": 0.03,
      "COVID": 0.02,
      "TB": 0.01
    }
  },
  "metadata": {
    "filename": "xray_20231225_153042.jpg",
    "processing_time_ms": 234
  }
}
```

### POST /receive
Uploads an image without classification

### GET /health
Health check endpoint

## Model Information

- **Architecture**: EfficientNet-B0
- **Classes**: NORMAL, PNEUMONIA, COVID, TB
- **Input Size**: 224x224 RGB
- **Normalization**: ImageNet statistics

## Files

- `app.py`: Main Flask application
- `model.py`: XRayClassifier model definition
- `inference.py`: Inference engine
- `requirements.txt`: Python dependencies
- `uploads/`: Upload directory (created automatically)

## Testing

Test the prediction endpoint:
```bash
curl -X POST http://localhost:5000/predict \
  -F "image=@path/to/xray.jpg" \
  -H "Content-Type: multipart/form-data"
```

## Notes

- The model loads on server startup (may take 10-30 seconds)
- CPU inference: ~2-5 seconds per image
- GPU inference: ~0.5-1 second per image
- Maximum file size: 16MB
