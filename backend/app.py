import time
import requests
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)

history_db = []
FLAGGED_WORDS = ["shocking", "secret", "miracle", "100% free", "you won't believe", "scandal", "leaked", "urgent", "click here", "overnight"]

# Load the trained Machine Learning models
print("Loading Machine Learning Models...")
try:
    model = joblib.load('model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    print("✅ Models Loaded Successfully!")
except Exception as e:
    print("❌ Error loading models. Please run train_model.py first.")

def extract_text_from_url(url):
    """Fetches and extracts paragraph text from a given URL."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text() for p in paragraphs])
        return text if text else "Error: Could not extract text."
    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/api/analyze', methods=['POST'])
def analyze_news():
    data = request.json
    input_data = data.get('text', '').strip()

    if not input_data:
        return jsonify({'error': 'No input provided'}), 400

    # Add a slight delay to simulate processing time
    time.sleep(1) 

    # Check if the input is a URL
    is_url = input_data.startswith('http://') or input_data.startswith('https://')
    
    if is_url:
        text_to_analyze = extract_text_from_url(input_data)
        if text_to_analyze.startswith("Error"):
            return jsonify({'error': 'Could not read the URL. Please paste the text directly.'}), 400
    else:
        text_to_analyze = input_data

    # Ensure text is long enough for accurate prediction
    if len(text_to_analyze) < 20:
        return jsonify({'error': 'Text is too short to analyze accurately.'}), 400

    # ----- Real AI Prediction Logic -----
    try:
        # Convert input text to vector format
        tfidf_text = vectorizer.transform([text_to_analyze])
        # Get probability of the text being real (index 1)
        probabilities = model.predict_proba(tfidf_text)[0] 
        real_probability = probabilities[1] 
        
        # Convert probability into a percentage score (0-100)
        score = int(real_probability * 100)
    except Exception as e:
        return jsonify({'error': 'Model prediction failed.'}), 500
    # ------------------------------------

    # Check for clickbait/flagged phrases
    found_flags = [word for word in FLAGGED_WORDS if word in text_to_analyze.lower()]

    # Determine verdict and UI color based on score
    if score > 70:
        verdict, color = "Highly Credible", "green"
    elif score > 40:
        verdict, color = "Suspicious", "yellow"
    else:
        verdict, color = "Likely Fake", "red"

    result = {
        'type': 'URL' if is_url else 'Text',
        'snippet': text_to_analyze[:150] + "...",
        'credibility_score': score,
        'flagged_words': found_flags,
        'verdict': verdict,
        'color': color,
        'timestamp': time.strftime("%Y-%m-%d %H:%M")
    }

    # Add result to history log (newest first)
    history_db.insert(0, result)

    # Return the result and the top 10 history items
    return jsonify({'result': result, 'history': history_db[:10]})

if __name__ == '__main__':
    app.run(debug=True, port=5000)