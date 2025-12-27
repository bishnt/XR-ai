import requests
import os
from PIL import Image
import io

# Create dummy image
img = Image.new('RGB', (100, 100), color = 'red')
img_byte_arr = io.BytesIO()
img.save(img_byte_arr, format='JPEG')
img_byte_arr = img_byte_arr.getvalue()

# Test /receive endpoint (simpler)
print("Testing /receive endpoint...")
try:
    files = {'image': ('test_image.jpg', img_byte_arr, 'image/jpeg')}
    response = requests.post('http://localhost:5000/receive', files=files)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error testing /receive: {e}")

# Test /predict endpoint
print("\nTesting /predict endpoint...")
try:
    # Re-create files dict because it might be consumed
    files = {'image': ('test_predict.jpg', img_byte_arr, 'image/jpeg')}
    response = requests.post('http://localhost:5000/predict', files=files)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error testing /predict: {e}")
