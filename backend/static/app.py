#!/usr/bin/env python3
"""
Flask Backend API for App Recommendation System
Serves ML model predictions and app recommendations
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import os
import json
from datetime import datetime
import logging
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from scipy.sparse import load_npz
from app_scraper import get_app_icons  # Custom scraper module
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

base_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.abspath(os.path.join(base_dir, "../templates"))
app = Flask(__name__, template_folder=template_dir)
CORS(app)  # Enable CORS for frontend integration

# Global variables for loaded enhanced models
apps_df = None
# Multiple similarity matrices
cosine_sim_tfidf = None
cosine_sim_ngram = None
linear_sim = None
cosine_sim_combined = None
# TF-IDF models
tfidf_basic = None
tfidf_ngram = None
tfidf_char = None
# Advanced ML models
lda_model = None
nmf_model = None
kmeans_model = None
scaler = None
rf_model = None
# Feature matrices and topics
lda_topics = None
nmf_features = None
count_vectorizer = None


def load_models():
    """Load the enhanced trained models and data"""
    global apps_df, cosine_sim_tfidf, cosine_sim_ngram, linear_sim, cosine_sim_combined
    global tfidf_basic, tfidf_ngram, tfidf_char, lda_model, nmf_model, kmeans_model
    global scaler, rf_model, lda_topics, nmf_features, count_vectorizer

    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.abspath(os.path.join(base_dir, "../../models"))

        # Load enhanced apps data
        apps_path = os.path.join(models_dir, "apps_list_enhanced.pkl")
        if os.path.exists(apps_path):
            apps_df = pickle.load(open(apps_path, "rb"))
            logger.info(f"Loaded {len(apps_df)} apps from enhanced dataset")
        else:
            # Fallback to original dataset
            apps_path = os.path.join(models_dir, "apps_list.pkl")
            if os.path.exists(apps_path):
                apps_df = pickle.load(open(apps_path, "rb"))
                logger.info(
                    f"Loaded {len(apps_df)} apps from original dataset")
            else:
                logger.error("No apps dataset found")
                return False

        # Load multiple similarity matrices
        similarity_files = [
            ("similarity_tfidf.pkl", "cosine_sim_tfidf"),
            ("similarity_ngram.pkl", "cosine_sim_ngram"),
            ("similarity_linear.pkl", "linear_sim"),
            ("similarity_combined.pkl", "cosine_sim_combined")
        ]

        for filename, var_name in similarity_files:
            filepath = os.path.join(models_dir, filename)
            if os.path.exists(filepath):
                globals()[var_name] = pickle.load(open(filepath, "rb"))
                logger.info(f"Loaded {var_name}: {globals()[var_name].shape}")
            else:
                logger.warning(f"Similarity matrix {filename} not found")

        # Load TF-IDF models
        tfidf_files = [
            ("tfidf_basic_model.pkl", "tfidf_basic"),
            ("tfidf_ngram_model.pkl", "tfidf_ngram"),
            ("tfidf_char_model.pkl", "tfidf_char")
        ]

        for filename, var_name in tfidf_files:
            filepath = os.path.join(models_dir, filename)
            if os.path.exists(filepath):
                globals()[var_name] = pickle.load(open(filepath, "rb"))
                logger.info(f"Loaded {var_name}")
            else:
                logger.warning(f"TF-IDF model {filename} not found")

        # Load advanced ML models
        ml_files = [
            ("lda_model.pkl", "lda_model"),
            ("nmf_model.pkl", "nmf_model"),
            ("kmeans_model.pkl", "kmeans_model"),
            ("scaler.pkl", "scaler"),
            ("rf_rating_model.pkl", "rf_model")
        ]

        for filename, var_name in ml_files:
            filepath = os.path.join(models_dir, filename)
            if os.path.exists(filepath):
                globals()[var_name] = pickle.load(open(filepath, "rb"))
                logger.info(f"Loaded {var_name}")
            else:
                logger.warning(f"ML model {filename} not found")

        # Load NumPy arrays
        lda_topics_path = os.path.join(models_dir, "lda_topics.npy")
        if os.path.exists(lda_topics_path):
            lda_topics = np.load(lda_topics_path)
            logger.info(f"Loaded LDA topics: {lda_topics.shape}")

        nmf_features_path = os.path.join(models_dir, "nmf_features.npy")
        if os.path.exists(nmf_features_path):
            nmf_features = np.load(nmf_features_path)
            logger.info(f"Loaded NMF features: {nmf_features.shape}")

        # Load count vectorizer vocabulary
        vocab_path = os.path.join(models_dir, "count_vectorizer_vocab.pkl")
        if os.path.exists(vocab_path):
            vocab = pickle.load(open(vocab_path, "rb"))
            count_vectorizer = CountVectorizer(vocabulary=vocab)
            logger.info("Re-created CountVectorizer from vocabulary")

        return True

    except Exception as e:
        logger.error(f"Error loading enhanced models: {e}")
        return False


def get_recommendations_by_title(title, method='tfidf_basic', num_recommendations=10):
    """Get recommendations based on app title using enhanced models"""
    try:
        if apps_df is None:
            return {"error": "Models not loaded"}

        # Find matching apps
        matching_apps = apps_df[apps_df['title'].str.contains(
            title, case=False, na=False)]

        if matching_apps.empty:
            return {"error": f"No app found with title containing: {title}"}

        idx = matching_apps.index[0]

        # Select similarity matrix based on method
        similarity_matrix = None
        if method == 'tfidf_basic' and cosine_sim_tfidf is not None:
            similarity_matrix = cosine_sim_tfidf
        elif method == 'tfidf_ngram' and cosine_sim_ngram is not None:
            similarity_matrix = cosine_sim_ngram
        elif method == 'linear' and linear_sim is not None:
            similarity_matrix = linear_sim
        elif method == 'combined' and cosine_sim_combined is not None:
            similarity_matrix = cosine_sim_combined
        elif cosine_sim_tfidf is not None:
            similarity_matrix = cosine_sim_tfidf  # Default fallback
        else:
            return {"error": "No similarity matrix available"}

        # Get similarity scores
        sim_scores = list(enumerate(similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        # Exclude the app itself
        sim_scores = sim_scores[1:num_recommendations+1]

        # Get app indices
        app_indices = [i[0] for i in sim_scores]

        # Create recommendations with enhanced features
        recommendations = []
        for i, app_idx in enumerate(app_indices):
            app = apps_df.iloc[app_idx]
            similarity_score = sim_scores[i][1]

            rec = {
                'appId': app['appId'],
                'title': app['title'],
                'genre': app['genre'],
                'developer': app['developer'],
                'summary': app['summary'] if 'summary' in app else "",
                'score': float(app['score']) if pd.notna(app['score']) else 0.0,
                'free': bool(app['free']),
                'price': float(app['price']) if pd.notna(app['price']) else 0.0,
                'ratings': int(app['ratings']) if pd.notna(app['ratings']) else 0,
                'similarity_score': float(similarity_score),
                'method': method
            }

            # Add enhanced features if available
            if 'rating_quality' in app:
                rec['rating_quality'] = float(app['rating_quality']) if pd.notna(
                    app['rating_quality']) else 0.0
            if 'popularity_score' in app:
                rec['popularity_score'] = float(app['popularity_score']) if pd.notna(
                    app['popularity_score']) else 0.0
            if 'cluster_label' in app:
                rec['cluster_label'] = int(app['cluster_label']) if pd.notna(
                    app['cluster_label']) else 0

            recommendations.append(rec)

        return {
            'query_app': title,
            'found_app': matching_apps.iloc[0]['title'],
            'method': method,
            'recommendations': recommendations
        }

    except Exception as e:
        logger.error(f"Error in get_recommendations_by_title: {e}")
        return {"error": str(e)}


def get_recommendations_hybrid(title, num_recommendations=10):
    """Get hybrid recommendations combining multiple methods"""
    try:
        if apps_df is None:
            return {"error": "Models not loaded"}

        # Get recommendations from different methods
        methods = []
        if cosine_sim_tfidf is not None:
            methods.append(('tfidf_basic', cosine_sim_tfidf, 0.3))
        if cosine_sim_ngram is not None:
            methods.append(('tfidf_ngram', cosine_sim_ngram, 0.4))
        if cosine_sim_combined is not None:
            methods.append(('combined', cosine_sim_combined, 0.3))

        if not methods:
            return {"error": "No similarity matrices available"}

        # Find the target app
        matching_apps = apps_df[apps_df['title'].str.contains(
            title, case=False, na=False)]
        if matching_apps.empty:
            return {"error": f"No app found with title containing: {title}"}

        idx = matching_apps.index[0]

        # Combine scores from different methods
        combined_scores = {}

        for method_name, similarity_matrix, weight in methods:
            sim_scores = list(enumerate(similarity_matrix[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

            # Get more to have options
            for app_idx, score in sim_scores[1:num_recommendations*2]:
                if app_idx not in combined_scores:
                    combined_scores[app_idx] = 0
                combined_scores[app_idx] += score * weight

        # Sort by combined score and get top recommendations
        top_indices = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[
            :num_recommendations]

        recommendations = []
        for app_idx, combined_score in top_indices:
            app = apps_df.iloc[app_idx]

            rec = {
                'appId': app['appId'],
                'title': app['title'],
                'genre': app['genre'],
                'developer': app['developer'],
                'summary': app['summary'] if 'summary' in app else "",
                'score': float(app['score']) if pd.notna(app['score']) else 0.0,
                'free': bool(app['free']),
                'price': float(app['price']) if pd.notna(app['price']) else 0.0,
                'ratings': int(app['ratings']) if pd.notna(app['ratings']) else 0,
                'similarity_score': float(combined_score),
                'method': 'hybrid'
            }

            # Add enhanced features if available
            if 'rating_quality' in app:
                rec['rating_quality'] = float(app['rating_quality']) if pd.notna(
                    app['rating_quality']) else 0.0
            if 'popularity_score' in app:
                rec['popularity_score'] = float(app['popularity_score']) if pd.notna(
                    app['popularity_score']) else 0.0

            recommendations.append(rec)

        return {
            'query_app': title,
            'found_app': matching_apps.iloc[0]['title'],
            'method': 'hybrid',
            'recommendations': recommendations
        }

    except Exception as e:
        logger.error(f"Error in get_recommendations_hybrid: {e}")
        return {"error": str(e)}


def get_recommendations_by_cluster(title, num_recommendations=10):
    """Get recommendations based on clustering"""
    try:
        if apps_df is None or 'cluster_label' not in apps_df.columns:
            return {"error": "Clustering model not available"}

        matching_apps = apps_df[apps_df['title'].str.contains(
            title, case=False, na=False)]
        if matching_apps.empty:
            return {"error": f"No app found with title containing: {title}"}

        app_cluster = matching_apps.iloc[0]['cluster_label']

        # Get other apps in the same cluster
        cluster_apps = apps_df[apps_df['cluster_label'] == app_cluster]
        cluster_apps = cluster_apps[~cluster_apps['title'].str.contains(
            title, case=False, na=False)]

        # Sort by enhanced criteria if available, otherwise by score
        if 'popularity_score' in cluster_apps.columns and 'rating_quality' in cluster_apps.columns:
            cluster_apps = cluster_apps.sort_values(
                ['popularity_score', 'rating_quality'], ascending=[False, False])
        else:
            cluster_apps = cluster_apps.sort_values(
                ['score', 'ratings'], ascending=[False, False])

        recommendations = []
        for _, app in cluster_apps.head(num_recommendations).iterrows():
            rec = {
                'appId': app['appId'],
                'title': app['title'],
                'genre': app['genre'],
                'developer': app['developer'],
                'summary': app['summary'] if 'summary' in app else "",
                'score': float(app['score']) if pd.notna(app['score']) else 0.0,
                'free': bool(app['free']),
                'price': float(app['price']) if pd.notna(app['price']) else 0.0,
                'ratings': int(app['ratings']) if pd.notna(app['ratings']) else 0,
                'cluster_label': int(app['cluster_label']),
                'method': 'cluster'
            }

            if 'rating_quality' in app:
                rec['rating_quality'] = float(app['rating_quality']) if pd.notna(
                    app['rating_quality']) else 0.0
            if 'popularity_score' in app:
                rec['popularity_score'] = float(app['popularity_score']) if pd.notna(
                    app['popularity_score']) else 0.0

            recommendations.append(rec)

        return {
            'query_app': title,
            'found_app': matching_apps.iloc[0]['title'],
            'cluster': int(app_cluster),
            'method': 'cluster',
            'recommendations': recommendations
        }

    except Exception as e:
        logger.error(f"Error in get_recommendations_by_cluster: {e}")
        return {"error": str(e)}


def get_recommendations_by_genre(genre, num_recommendations=10):
    """Get enhanced recommendations by genre"""
    try:
        if apps_df is None:
            return {"error": "Models not loaded"}

        genre_apps = apps_df[apps_df['genre'].str.contains(
            genre, case=False, na=False)]

        if genre_apps.empty:
            return {"error": f"No apps found in genre: {genre}"}

        # Sort by enhanced criteria if available, otherwise by score and ratings
        if 'rating_quality' in genre_apps.columns and 'popularity_score' in genre_apps.columns:
            genre_apps_sorted = genre_apps.sort_values(
                ['rating_quality', 'popularity_score'], ascending=[False, False])
        else:
            genre_apps_sorted = genre_apps.sort_values(
                ['score', 'ratings'], ascending=[False, False])

        top_apps = genre_apps_sorted.head(num_recommendations)

        recommendations = []
        for _, app in top_apps.iterrows():
            rec = {
                'appId': app['appId'],
                'title': app['title'],
                'genre': app['genre'],
                'developer': app['developer'],
                'summary': app['summary'] if 'summary' in app else "",
                'score': float(app['score']) if pd.notna(app['score']) else 0.0,
                'free': bool(app['free']),
                'price': float(app['price']) if pd.notna(app['price']) else 0.0,
                'ratings': int(app['ratings']) if pd.notna(app['ratings']) else 0
            }

            # Add enhanced features if available
            if 'rating_quality' in app:
                rec['rating_quality'] = float(app['rating_quality']) if pd.notna(
                    app['rating_quality']) else 0.0
            if 'popularity_score' in app:
                rec['popularity_score'] = float(app['popularity_score']) if pd.notna(
                    app['popularity_score']) else 0.0

            recommendations.append(rec)

        return {
            'query_genre': genre,
            'recommendations': recommendations
        }

    except Exception as e:
        logger.error(f"Error in get_recommendations_by_genre: {e}")
        return {"error": str(e)}


def get_popular_apps(num_recommendations=10):
    """Get most popular apps with enhanced scoring"""
    try:
        if apps_df is None:
            return {"error": "Models not loaded"}

        # Use enhanced popularity scoring if available
        if 'rating_quality' in apps_df.columns and 'popularity_score' in apps_df.columns:
            # Create enhanced popularity score
            apps_copy = apps_df.copy()
            apps_copy['enhanced_popularity'] = (
                apps_copy['score'] * 0.3 +
                apps_copy['rating_quality'] * 0.4 +
                apps_copy['popularity_score'] * 0.3
            )
            popular_apps = apps_copy.sort_values(
                'enhanced_popularity', ascending=False)
        else:
            popular_apps = apps_df.sort_values(
                ['score', 'ratings'], ascending=[False, False])

        top_apps = popular_apps.head(num_recommendations)

        recommendations = []
        for _, app in top_apps.iterrows():
            rec = {
                'appId': app['appId'],
                'title': app['title'],
                'genre': app['genre'],
                'developer': app['developer'],
                'summary': app['summary'] if 'summary' in app else "",
                'score': float(app['score']) if pd.notna(app['score']) else 0.0,
                'free': bool(app['free']),
                'price': float(app['price']) if pd.notna(app['price']) else 0.0,
                'ratings': int(app['ratings']) if pd.notna(app['ratings']) else 0
            }

            # Add enhanced features if available
            if 'rating_quality' in app:
                rec['rating_quality'] = float(app['rating_quality']) if pd.notna(
                    app['rating_quality']) else 0.0
            if 'popularity_score' in app:
                rec['popularity_score'] = float(app['popularity_score']) if pd.notna(
                    app['popularity_score']) else 0.0
            if 'enhanced_popularity' in app:
                rec['enhanced_popularity'] = float(app['enhanced_popularity']) if pd.notna(
                    app['enhanced_popularity']) else 0.0

            recommendations.append(rec)

        return {
            'type': 'popular',
            'recommendations': recommendations
        }

    except Exception as e:
        logger.error(f"Error in get_popular_apps: {e}")
        return {"error": str(e)}

def add_icons_to_apps(app_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Fungsi jembatan untuk mengambil ikon aplikasi menggunakan scraper dan menambahkannya ke data.
    """
    if not app_data:
        return []

    # 1. Siapkan data untuk scraper. Scraper Anda membutuhkan 'app_id' dan 'title'.
    apps_to_fetch = []
    for app in app_data:
        # Pastikan data memiliki 'appId' dan 'title'
        # Proyek teman Anda sepertinya sudah konsisten menggunakan 'title'.
        # Kita perlu memastikan 'appId' ada.
        if 'appId' in app and 'title' in app:
            apps_to_fetch.append({
                'app_id': app['appId'], # Scraper Anda butuh 'app_id'
                'title': app['title']
            })

    if not apps_to_fetch:
        # Jika tidak ada data valid, kembalikan data asli
        return app_data

    # 2. Panggil scraper dengan data yang sudah disiapkan
    # get_app_icons akan mengembalikan map: {'com.whatsapp': 'url_gambar', ...}
    logging.info(f"Fetching icons for {len(apps_to_fetch)} apps...")
    icon_urls_map = get_app_icons(apps_to_fetch)
    logging.info("Icon fetching complete.")
    
    # 3. Update data aplikasi asli dengan URL ikon yang didapat
    for app in app_data:
        app_id = app.get('appId')
        if app_id and app_id in icon_urls_map:
            app['icon_url'] = icon_urls_map[app_id]
        else:
            # Pastikan ada field 'icon_url' meskipun gagal, untuk konsistensi di frontend
            app['icon_url'] = None
            
    return app_data

# API Routes
@app.route('/')
def home():
    """Home page"""
    return render_template('index.html')


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    models_loaded = (apps_df is not None and
                     (cosine_sim_tfidf is not None or
                      cosine_sim_ngram is not None or
                      linear_sim is not None))

    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': models_loaded,
        'available_methods': {
            'tfidf_basic': cosine_sim_tfidf is not None,
            'tfidf_ngram': cosine_sim_ngram is not None,
            'linear_similarity': linear_sim is not None,
            'combined_features': cosine_sim_combined is not None,
            'clustering': kmeans_model is not None,
            'topic_modeling': lda_model is not None,
            'rating_prediction': rf_model is not None
        }
    })


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get enhanced dataset statistics"""
    try:
        if apps_df is None:
            return jsonify({"error": "Models not loaded"}), 500

        stats = {
            'total_apps': len(apps_df),
            'free_apps': len(apps_df[apps_df['free'] == 1]),
            'paid_apps': len(apps_df[apps_df['free'] == 0]),
            'avg_score': float(apps_df['score'].mean()),
            'avg_ratings': float(apps_df['ratings'].mean()),
            'top_genres': apps_df['genre'].value_counts().head(5).to_dict(),
            'score_range': {
                'min': float(apps_df['score'].min()),
                'max': float(apps_df['score'].max())
            }
        }

        # Add enhanced stats if available
        if 'rating_quality' in apps_df.columns:
            stats['avg_rating_quality'] = float(
                apps_df['rating_quality'].mean())
        if 'popularity_score' in apps_df.columns:
            stats['avg_popularity_score'] = float(
                apps_df['popularity_score'].mean())
        if 'cluster_label' in apps_df.columns:
            stats['num_clusters'] = int(apps_df['cluster_label'].nunique())
        if 'price_category' in apps_df.columns:
            stats['price_categories'] = apps_df['price_category'].value_counts().to_dict()

        return jsonify(stats)

    except Exception as e:
        logger.error(f"Error in get_stats: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/recommend/title', methods=['POST'])
def recommend_by_title():
    """Get recommendations based on app title with method selection"""
    try:
        data = request.get_json()

        if not data or 'title' not in data:
            return jsonify({"error": "Title is required"}), 400

        title = data['title']
        num_recommendations = data.get('num_recommendations', 10)
        method = data.get('method', 'tfidf_basic')  # Default method

        if num_recommendations > 50:
            num_recommendations = 50

        result = get_recommendations_by_title(
            title, method, num_recommendations)

        if 'error' in result:
            return jsonify(result), 404

        # --- INTEGRASI DIMULAI DI SINI ---
        # Ambil daftar rekomendasi dari hasil
        recommendations_list = result.get('recommendations', [])
        
        # Panggil fungsi jembatan kita untuk menambahkan URL ikon
        recommendations_with_icons = add_icons_to_apps(recommendations_list)
        
        # Masukkan kembali daftar yang sudah diperbarui ke dalam hasil
        result['recommendations'] = recommendations_with_icons
        # --- INTEGRASI SELESAI ---

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in recommend_by_title: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/recommend/hybrid', methods=['POST'])
def recommend_hybrid():
    """Get hybrid recommendations combining multiple methods"""
    try:
        data = request.get_json()

        if not data or 'title' not in data:
            return jsonify({"error": "Title is required"}), 400

        title = data['title']
        num_recommendations = data.get('num_recommendations', 10)

        if num_recommendations > 50:
            num_recommendations = 50

        result = get_recommendations_hybrid(title, num_recommendations)

        if 'error' in result:
            return jsonify(result), 404

        # --- INTEGRASI DIMULAI DI SINI ---
        # Ambil daftar rekomendasi dari hasil
        recommendations_list = result.get('recommendations', [])
        
        # Panggil fungsi jembatan kita untuk menambahkan URL ikon
        recommendations_with_icons = add_icons_to_apps(recommendations_list)
        
        # Masukkan kembali daftar yang sudah diperbarui ke dalam hasil
        result['recommendations'] = recommendations_with_icons
        # --- INTEGRASI SELESAI ---
        
        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in recommend_hybrid: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/recommend/cluster', methods=['POST'])
def recommend_by_cluster():
    """Get recommendations based on clustering"""
    try:
        data = request.get_json()

        if not data or 'title' not in data:
            return jsonify({"error": "Title is required"}), 400

        title = data['title']
        num_recommendations = data.get('num_recommendations', 10)

        if num_recommendations > 50:
            num_recommendations = 50

        result = get_recommendations_by_cluster(title, num_recommendations)

        if 'error' in result:
            return jsonify(result), 404
        
        # --- INTEGRASI DIMULAI DI SINI ---
        # Ambil daftar rekomendasi dari hasil
        recommendations_list = result.get('recommendations', [])
        
        # Panggil fungsi jembatan kita untuk menambahkan URL ikon
        recommendations_with_icons = add_icons_to_apps(recommendations_list)
        
        # Masukkan kembali daftar yang sudah diperbarui ke dalam hasil
        result['recommendations'] = recommendations_with_icons
        # --- INTEGRASI SELESAI ---

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in recommend_by_cluster: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/recommend/genre', methods=['POST'])
def recommend_by_genre():
    """Get recommendations based on genre"""
    try:
        data = request.get_json()

        if not data or 'genre' not in data:
            return jsonify({"error": "Genre is required"}), 400

        genre = data['genre']
        num_recommendations = data.get('num_recommendations', 10)

        if num_recommendations > 50:
            num_recommendations = 50

        result = get_recommendations_by_genre(genre, num_recommendations)

        if 'error' in result:
            return jsonify(result), 404
        
        # --- INTEGRASI DIMULAI DI SINI ---
        # Ambil daftar rekomendasi dari hasil
        recommendations_list = result.get('recommendations', [])
        
        # Panggil fungsi jembatan kita untuk menambahkan URL ikon
        recommendations_with_icons = add_icons_to_apps(recommendations_list)
        
        # Masukkan kembali daftar yang sudah diperbarui ke dalam hasil
        result['recommendations'] = recommendations_with_icons
        # --- INTEGRASI SELESAI ---

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in recommend_by_genre: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/recommend/popular', methods=['GET'])
def recommend_popular():
    """Get popular apps"""
    try:
        num_recommendations = request.args.get(
            'num_recommendations', 10, type=int)

        if num_recommendations > 50:
            num_recommendations = 50

        result = get_popular_apps(num_recommendations)

        if 'error' in result:
            return jsonify(result), 500
        
        # --- INTEGRASI DIMULAI DI SINI ---
        # Ambil daftar rekomendasi dari hasil
        recommendations_list = result.get('recommendations', [])
        
        # Panggil fungsi jembatan kita untuk menambahkan URL ikon
        recommendations_with_icons = add_icons_to_apps(recommendations_list)
        
        # Masukkan kembali daftar yang sudah diperbarui ke dalam hasil
        result['recommendations'] = recommendations_with_icons
        # --- INTEGRASI SELESAI ---

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in recommend_popular: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/methods', methods=['GET'])
def get_available_methods():
    """Get list of available recommendation methods"""
    try:
        methods = {
            'similarity_based': [],
            'clustering': kmeans_model is not None,
            'topic_modeling': lda_model is not None,
            'hybrid': False
        }

        if cosine_sim_tfidf is not None:
            methods['similarity_based'].append('tfidf_basic')
        if cosine_sim_ngram is not None:
            methods['similarity_based'].append('tfidf_ngram')
        if linear_sim is not None:
            methods['similarity_based'].append('linear')
        if cosine_sim_combined is not None:
            methods['similarity_based'].append('combined')

        # Hybrid is available if we have multiple similarity methods
        methods['hybrid'] = len(methods['similarity_based']) > 1

        return jsonify(methods)

    except Exception as e:
        logger.error(f"Error in get_available_methods: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/genres', methods=['GET'])
def get_genres():
    """Get list of available genres"""
    try:
        if apps_df is None:
            return jsonify({"error": "Models not loaded"}), 500

        genres = apps_df['genre'].value_counts().to_dict()
        genre_list = [{"name": genre, "count": count}
                      for genre, count in genres.items()]

        return jsonify({"genres": genre_list})

    except Exception as e:
        logger.error(f"Error in get_genres: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/search', methods=['POST'])
def search_apps():
    """Search apps by title"""
    try:
        data = request.get_json()

        if not data or 'query' not in data:
            return jsonify({"error": "Search query is required"}), 400

        query = data['query']
        limit = data.get('limit', 20)

        if apps_df is None:
            return jsonify({"error": "Models not loaded"}), 500

        # Search in titles
        matches = apps_df[apps_df['title'].str.contains(
            query, case=False, na=False)]
        matches = matches.head(limit)

        results = []
        for _, app in matches.iterrows():
            result = {
                'title': app['title'],
                'genre': app['genre'],
                'developer': app['developer'],
                'score': float(app['score']) if pd.notna(app['score']) else 0.0,
                'free': bool(app['free'])
            }
            results.append(result)

        return jsonify({
            'query': query,
            'results': results,
            'total_found': len(matches)
        })

    except Exception as e:
        logger.error(f"Error in search_apps: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/search/enhanced', methods=['POST'])
def search_apps_enhanced():
    """Enhanced search with multiple criteria"""
    try:
        data = request.get_json()

        if not data or 'query' not in data:
            return jsonify({"error": "Search query is required"}), 400

        query = data['query']
        limit = data.get('limit', 20)
        search_fields = data.get(
            'search_fields', ['title', 'genre', 'developer'])

        if apps_df is None:
            return jsonify({"error": "Models not loaded"}), 500

        # Build search mask based on specified fields
        mask = pd.Series([False] * len(apps_df))

        if 'title' in search_fields:
            mask |= apps_df['title'].str.contains(query, case=False, na=False)
        if 'genre' in search_fields:
            mask |= apps_df['genre'].str.contains(query, case=False, na=False)
        if 'developer' in search_fields:
            mask |= apps_df['developer'].str.contains(
                query, case=False, na=False)
        if 'summary' in search_fields and 'summary' in apps_df.columns:
            mask |= apps_df['summary'].str.contains(
                query, case=False, na=False)

        matches = apps_df[mask]

        # Sort by enhanced criteria if available
        if 'rating_quality' in matches.columns:
            matches = matches.sort_values('rating_quality', ascending=False)
        else:
            matches = matches.sort_values('score', ascending=False)

        matches = matches.head(limit)

        results = []
        for _, app in matches.iterrows():
            result = {
                'appId': app['appId'],
                'title': app['title'],
                'genre': app['genre'],
                'developer': app['developer'],
                'summary': app['summary'] if 'summary' in app else "",
                'score': float(app['score']) if pd.notna(app['score']) else 0.0,
                'free': bool(app['free']),
                'price': float(app['price']) if pd.notna(app['price']) else 0.0,
                'ratings': int(app['ratings']) if pd.notna(app['ratings']) else 0
            }

            # Add enhanced features if available
            if 'rating_quality' in app:
                result['rating_quality'] = float(app['rating_quality']) if pd.notna(
                    app['rating_quality']) else 0.0
            if 'popularity_score' in app:
                result['popularity_score'] = float(app['popularity_score']) if pd.notna(
                    app['popularity_score']) else 0.0

            results.append(result)

        return jsonify({
            'query': query,
            'search_fields': search_fields,
            'results': results,
            'total_found': len(matches)
        })

    except Exception as e:
        logger.error(f"Error in search_apps_enhanced: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Load enhanced models on startup
    logger.info("Starting Enhanced App Recommendation System...")
    logger.info("Loading enhanced models...")

    if load_models():
        logger.info("Enhanced models loaded successfully!")
        logger.info("Available features:")
        logger.info(f"- Enhanced dataset: {apps_df is not None}")
        logger.info(f"- TF-IDF Basic: {cosine_sim_tfidf is not None}")
        logger.info(f"- TF-IDF N-gram: {cosine_sim_ngram is not None}")
        logger.info(f"- Linear Similarity: {linear_sim is not None}")
        logger.info(f"- Combined Features: {cosine_sim_combined is not None}")
        logger.info(f"- Clustering: {kmeans_model is not None}")
        logger.info(f"- Topic Modeling: {lda_model is not None}")
        logger.info(f"- Rating Prediction: {rf_model is not None}")
        logger.info("Starting Flask server...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        logger.error("Failed to load enhanced models.")
        logger.error(
            "Please run the enhanced spark.py first to generate model files.")
        logger.error(
            "Required files: apps_list_enhanced.pkl, similarity_*.pkl, tfidf_*.pkl")
