# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Install system dependencies for OpenCV and other libraries
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
# Using the CPU-only version of torch to keep the image size smaller and compatible with free tier CPU
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy the rest of the application code
COPY . .

# Create the uploads directory if it doesn't exist
RUN mkdir -p uploads

# Expose the port the app runs on
EXPOSE 7860

# Set environment variables
ENV FLASK_APP=app.py
ENV PORT=7860

# Run the application
# Hugging Face Spaces expect the application to run on port 7860
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--timeout", "120", "app:app"]
