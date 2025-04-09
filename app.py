from flask import Flask, request, jsonify
import requests
import numpy as np
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.tokenize import sent_tokenize
import wikipedia
import re
from dotenv import load_dotenv
import os

load_dotenv()
# Initialize Flask app
app = Flask(__name__)

# Download NLTK data (needed for sent_tokenize)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class EnhancedFactChecker:
    def __init__(self, google_api_key):
        self.GOOGLE_API_KEY = google_api_key
        self.FACTCHECK_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def clean_text(self, text):
        """Clean text for comparison"""
        text = re.sub(r'\[\d+\]|\[citation needed\]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def fetch_fact_check(self, query):
        """Fetch results from Google Fact Check API"""
        params = {
            "query": query[:200],  # Limit query length
            "key": self.GOOGLE_API_KEY,
            "languageCode": "en"
        }
        try:
            response = requests.get(self.FACTCHECK_URL, params=params)
            if response.status_code == 200:
                data = response.json()
                if "claims" in data:
                    return [
                        {
                            "text": claim["text"],
                            "verdict": claim["claimReview"][0]["textualRating"],
                            "source": claim["claimReview"][0]["url"]
                        }
                        for claim in data["claims"]
                    ]
        except Exception as e:
            print(f"Google Fact Check API error: {e}")
        return []

    def check_wikipedia(self, text):
        """Search Wikipedia for relevant information"""
        try:
            # Extract first sentence for search
            first_sentence = sent_tokenize(text)[0]
            keywords = ' '.join(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', first_sentence))

            if not keywords:
                keywords = ' '.join(first_sentence.split()[:5])

            search_results = wikipedia.search(keywords, results=1)

            if search_results:
                page = wikipedia.page(search_results[0], auto_suggest=False)
                return {
                    "text": page.summary,
                    "source": page.url
                }
        except Exception as e:
            print(f"Wikipedia error: {e}")
        return None

    def calculate_similarity(self, text1, text2):
        """Calculate semantic similarity between texts"""
        try:
            embedding1 = self.model.encode(text1, convert_to_tensor=True)
            embedding2 = self.model.encode(text2, convert_to_tensor=True)
            return float(util.pytorch_cos_sim(embedding1, embedding2))
        except:
            return 0

    def process_fact_checking_results(self, text):
        """Process text using both Google Fact Check and Wikipedia"""
        cleaned_text = self.clean_text(text)

        # Get fact check results
        fact_results = self.fetch_fact_check(cleaned_text)

        # Get Wikipedia results
        wiki_result = self.check_wikipedia(cleaned_text)

        # Calculate similarities and confidence
        confidences = []

        # Check Google Fact Check results
        if fact_results:
            for fact in fact_results:
                similarity = self.calculate_similarity(cleaned_text, fact["text"])
                confidences.append(similarity)

        # Check Wikipedia result
        if wiki_result:
            wiki_similarity = self.calculate_similarity(cleaned_text, wiki_result["text"])
            confidences.append(wiki_similarity)

        # Calculate final confidence
        confidence = int(np.mean(confidences) * 100) if confidences else 0

        # Determine result based on available information
        if fact_results or wiki_result:
            if confidence >= 70:
                alert = "True"
            elif confidence >= 50:
                alert = "Partially True"
            else:
                alert = "Needs Verification"

            # Use most relevant source for summary
            if wiki_result:
                summary = wiki_result["text"]
                source = wiki_result["source"]
            else:
                summary = fact_results[0]["text"]
                source = fact_results[0]["source"]
        else:
            alert = "No Data"
            confidence = 0
            summary = "No relevant information found"
            source = None

        return {
            "alert": alert,
            "confidence": confidence,
            "summary": summary,
            "source": source
        }

# Create an instance of EnhancedFactChecker with the API key
# Replace with your actual API key
GOOGLE_API_KEY =os.getenv("GOOGLE_API_KEY")
fact_checker = EnhancedFactChecker(GOOGLE_API_KEY)

# API Endpoints
@app.route('/api/fact-check', methods=['POST'])
def fact_check():
    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({
            'error': 'Missing required parameter: text'
        }), 400
    
    text = data['text']
    result = fact_checker.process_fact_checking_results(text)
    
    return jsonify(result)

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'message': 'Enhanced Fact-checking API is running'
    })

# Error handling
@app.errorhandler(Exception)
def handle_error(e):
    return jsonify({
        'error': str(e),
        'message': 'An error occurred processing the request'
    }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)