# Flask Backend - Image Upload API

A simple Flask backend for receiving and processing image uploads from the frontend.

## Features

- **Image Upload**: Accepts image files via POST request
- **CORS Support**: Enabled for frontend communication
- **File Validation**: Validates file types and sizes
- **Unique Filenames**: Generates timestamped filenames to avoid collisions
- **Error Handling**: Comprehensive error responses
- **Health Check**: Endpoint to verify server status

## Setup

### 1. Create a Virtual Environment (recommended)

```bash
python -m venv venv
```

### 2. Activate Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Server

```bash
python app.py
```

The server will start on `http://localhost:5000`

## API Endpoints

### POST `/receive`

Receives and stores uploaded images.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: Form data with `image` field containing the file

**Success Response (200):**
```json
{
  "success": true,
  "message": "Image received successfully",
  "data": {
    "filename": "image_20251225_155830.jpg",
    "original_filename": "image.jpg",
    "filepath": "uploads/image_20251225_155830.jpg",
    "file_size": 1048576,
    "file_size_mb": 1.0,
    "timestamp": "20251225_155830"
  }
}
```

**Error Response (400):**
```json
{
  "success": false,
  "error": "Invalid file type",
  "message": "Allowed file types: png, jpg, jpeg, gif, bmp, tiff, webp",
  "filename": "document.pdf"
}
```

### GET `/health`

Health check endpoint to verify server status.

**Success Response (200):**
```json
{
  "status": "healthy",
  "message": "Flask server is running",
  "upload_folder": "uploads",
  "max_file_size_mb": 16.0
}
```

## Configuration

- **Upload Folder**: `uploads/`
- **Max File Size**: 16 MB
- **Allowed Extensions**: png, jpg, jpeg, gif, bmp, tiff, webp
- **Port**: 5000
- **Host**: 0.0.0.0 (accessible from network)

## Frontend Integration Example

```javascript
const formData = new FormData();
formData.append('image', imageFile);

const response = await fetch('http://localhost:5000/receive', {
  method: 'POST',
  body: formData,
});

const result = await response.json();
console.log(result);
```
