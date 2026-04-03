# Fake News Detector

Full-stack web app to detect fake news using ML model.

## Tech Stack
- **Backend**: Flask (Python), scikit-learn TF-IDF + PassiveAggressiveClassifier
- **Frontend**: React + Tailwind CSS

## Setup

### Backend
```bash
cd backend
pip install -r requirements.txt  # Create if needed: flask, scikit-learn, pandas, joblib
python train_model.py  # Train model (generates model.pkl, vectorizer.pkl)
python app.py
```
Backend runs on http://localhost:5000

### Frontend
```bash
cd frontend
npm install
npm start
```
Frontend runs on http://localhost:3000 (proxies /api to backend)

## API
- POST /predict: { "text": "news text" } → { "prediction": "FAKE"/"REAL", "probability": float }

## Model Accuracy
~99% on test set (binary classification on True/Fake datasets)
