# xrAI - Medical X-Ray Analysis

xrAI is a modern web application designed for automated classification of chest X-rays. It uses deep learning to identify conditions such as Normal, Pneumonia, and TB.

## Features
- **Deep Learning Inference**: Fast and accurate X-ray classification.
- **Grad-CAM Visualization**: View heatmaps showing which areas the AI focused on.
- **Modern UI**: Clean, clinical-grade interface built with Next.js.

## Tech Stack
- **Frontend**: Next.js (React), TailwindCSS, Axios
- **Backend**: Flask (Python), PyTorch, OpenCV

---

## Getting Started

### Backend Setup
1. Navigate to the `backend` directory:
   ```bash
   cd backend
   ```
2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file from the example:
   ```bash
   cp .env.example .env
   ```
5. Run the server:
   ```bash
   python app.py
   ```

### Frontend Setup
1. Navigate to the `frontend/my-app` directory:
   ```bash
   cd frontend/my-app
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Create a `.env.local` file:
   ```bash
   NEXT_PUBLIC_API_URL=http://localhost:5000
   ```
4. Run the development server:
   ```bash
   npm run dev
   ```

---

## Deployment

### Backend (Render)
This project is configured for deployment on [Render](https://render.com).
- Use the `backend/render.yaml` blueprint.
- Ensure you have the `best_xray_model.pth` file in the `backend` directory.

### Frontend (Vercel/Netlify)
- Set the `NEXT_PUBLIC_API_URL` environment variable to your deployed backend URL.

---

## Environment Variables

### Backend
- `PORT`: The port the Flask server will run on (default: 5000).

### Frontend
- `NEXT_PUBLIC_API_URL`: The URL of the Flask backend.
