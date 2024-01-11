import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request, jsonify
from fuzzywuzzy import fuzz
import logging
import json
from flask_cors import CORS  # Import CORS


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

dataset_path = "resturants.csv"
df = pd.read_csv(dataset_path)

    # Assuming 'df' is your DataFrame with 'name', 'cuisine1', 'cuisine2', 'cuisine3' columns
features = df[['cuisine1', 'cuisine2', 'cuisine3']].astype(str).apply(lambda x: ' '.join(x), axis=1).tolist()

    #Create the TfidfVectorizer with multiple stop words for both Arabic and English
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(features)
similarity_matrix = cosine_similarity(tfidf_matrix)

    # Create indices dictionary
indices = pd.Series(df.index, index=df['name']).to_dict()
    
def restaurant_recommendation(name, similarity_matrix=similarity_matrix, indices=indices):
    name = name.lower()

    if name not in indices:
        return {'error': f"Error: {name} not found in the indices dictionary."}

    index = indices[name]
    similarity_scores = list(enumerate(similarity_matrix[index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Exclude the input restaurant itself
    similarity_scores = similarity_scores[1:11] if len(similarity_scores) > 1 else similarity_scores

    restaurant_indices = [i[0] for i in similarity_scores]

    recommendations = []
    for _, row in df.iloc[restaurant_indices].iterrows():
        recommendation = {
            'name': row['name'],
            'address': row['address'],
            'price': row['price'],
            'rating': row['rating'],
            'cuisine1': row['cuisine1'],
            'cuisine2': row['cuisine2'] if not pd.isna(row['cuisine2']) else None,
            'cuisine3': row['cuisine3'] if not pd.isna(row['cuisine3']) else None
        }

        recommendations.append(recommendation)

    return recommendations

@app.route('/get_recommendation', methods=['POST'])
def recommend():
    try:
        user_input = request.json.get('input_text', '')
        app.logger.info(f"Received request with input: {user_input}")

        # Call the restaurant_recommendation function with similarity_matrix and indices arguments
        recommendation = restaurant_recommendation(user_input)

        app.logger.info(f"Recommendation: {recommendation}")

        # Return the recommendation as a JSON response
        return jsonify({'recommendation': recommendation})

    except Exception as e:
        app.logger.error(f"Error: {e}")
        return jsonify({'error': str(e)}), 500  # Return a 500 status code for an internal server error

if __name__ == '__main__':
    app.run(debug=True)
    
@app.route('/')
def index():
    return render_template('index.html')
    



