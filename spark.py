#!/usr/bin/env python
# coding: utf-8

from scipy.sparse import hstack, csr_matrix
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.feature import CountVectorizer, RegexTokenizer, StopWordsRemover, HashingTF, IDF, VectorAssembler
from pyspark.sql.functions import concat_ws, split, col, when, isnan, isnull, regexp_replace, lower, trim
from pyspark.sql.types import StringType, FloatType, IntegerType
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer as SKCountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import pickle
import os
import json
import re
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

# Create a SparkSession
# NEW, SELF-CONTAINED CODE
spark = SparkSession.builder \
    .appName("AppRecommendationSystem") \
    .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4,io.delta:delta-core_2.12:2.4.0") \
    .config("spark.hadoop.fs.s3a.access.key", "minio_access_key") \
    .config("spark.hadoop.fs.s3a.secret.key", "minio_secret_key") \
    .config("spark.hadoop.fs.s3a.endpoint", "http://localhost:9000") \
    .config("spark.hadoop.fs.s3a.path.style.access", "true") \
    .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .getOrCreate()

# Read batched data from S3 bucket (apps data)
apps_df = spark.read.csv("s3a://apps/*.csv", header=True, inferSchema=True)

# Advanced Data Cleaning and Preprocessing
print("=== Data Cleaning ===")

# Check for null values in each column
print("Null values per column:")
for col_name in apps_df.columns:
    null_count = apps_df.filter(col(col_name).isNull()).count()
    print(f"{col_name}: {null_count}")

# Enhanced null value handling
apps_df = apps_df.fillna({
    "summary": "No description available",
    "genre": "Unknown",
    "title": "Unknown App",
    "developer": "Unknown Developer",
    "score": 0.0,
    "ratings": 0,
    "price": 0.0,
    "free": 1,  # Default to free if unknown
    "minInstalls": 0,
    "reviews": 0
})

# Clean text fields - remove special characters and normalize
apps_df = apps_df.withColumn("summary",
                             regexp_replace(col("summary"), "[^a-zA-Z0-9\\s]", " "))
apps_df = apps_df.withColumn("summary",
                             regexp_replace(col("summary"), "\\s+", " "))
apps_df = apps_df.withColumn("summary", trim(lower(col("summary"))))

apps_df = apps_df.withColumn("genre", trim(col("genre")))
apps_df = apps_df.withColumn("developer", trim(col("developer")))
apps_df = apps_df.withColumn("title", trim(col("title")))

# Handle outliers in numerical fields
print("=== Handling Outliers ===")
# Cap extremely high scores
apps_df = apps_df.withColumn("score",
                             when(col("score") > 5.0, 5.0)
                             .when(col("score") < 0.0, 0.0)
                             .otherwise(col("score")))

# Cap extremely high prices (likely errors)
apps_df = apps_df.withColumn("price",
                             when(col("price") > 500.0, 500.0)
                             .when(col("price") < 0.0, 0.0)
                             .otherwise(col("price")))

# Remove rows where essential fields are missing even after filling
apps_df = apps_df.filter(
    (col("title").isNotNull()) &
    (col("title") != "") &
    (col("title") != "Unknown App") &
    (col("appId").isNotNull()) &
    (col("summary") != "") &
    (col("summary") != "No description available")
)

print(f"Records after cleaning: {apps_df.count()}")

# Feature engineering - select relevant columns for recommendation
print("=== Feature Engineering ===")
apps_df = apps_df.select("appId", "title", "summary", "genre", "developer",
                         "score", "free", "price", "ratings", "minInstalls", "reviews")

# Create additional features
print("Creating additional features...")

# Text length features
apps_df = apps_df.withColumn("summary_length",
                             when(col("summary").isNotNull(),
                                  regexp_replace(col("summary"), "\\s+", " ").rlike(".*")).cast("int"))

# Price categories
apps_df = apps_df.withColumn("price_category",
                             when(col("price") == 0, "free")
                             .when(col("price") <= 2.99, "cheap")
                             .when(col("price") <= 9.99, "mid")
                             .otherwise("premium"))

# Rating quality score (combination of score and number of ratings)
apps_df = apps_df.withColumn("rating_quality",
                             col("score") * (col("ratings") / (col("ratings") + 100)))

# Popularity score based on installs and ratings
apps_df = apps_df.withColumn("popularity_score",
                             (col("minInstalls") / 1000000) * 0.3 +
                             (col("ratings") / 100000) * 0.4 +
                             col("score") * 0.3)

# Tokenize the summary text data with enhanced preprocessing
print("=== Text Processing with Multiple Approaches ===")
tokenizer = RegexTokenizer(
    inputCol="summary", outputCol="words", pattern="\\W", minTokenLength=2)
apps_df = tokenizer.transform(apps_df)

# Remove stop words from summary
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
apps_df = remover.transform(apps_df)

# Create comprehensive tags including all text features
apps_df = apps_df.withColumn("comprehensive_tags", concat_ws(" ",
                                                             apps_df["filtered_words"],
                                                             apps_df["genre"],
                                                             apps_df["developer"],
                                                             apps_df["price_category"]
                                                             ))

# Convert to array format
apps_df = apps_df.withColumn("comprehensive_tags", split(
    apps_df["comprehensive_tags"], " "))

# Clean up intermediate columns but keep important ones
apps_df = apps_df.drop("words")

print("=== Feature Engineering Complete ===")
apps_df.select("appId", "title", "genre", "developer", "summary",
               "comprehensive_tags", "rating_quality", "popularity_score").show(5, truncate=False)

# Multiple Vectorization Approaches
print("=== Creating Multiple Feature Vectors ===")

# 1. Count Vectorizer (existing approach)
print("1. Count Vectorizer...")
cv = CountVectorizer(inputCol="comprehensive_tags", outputCol="count_features",
                     vocabSize=10000, minDF=2)
count_model = cv.fit(apps_df)
apps_df = count_model.transform(apps_df)

# 2. TF-IDF Vectorizer using Spark ML
print("2. TF-IDF Vectorizer...")

# Use HashingTF instead of TF. It takes the "filtered_words" column directly.
hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=10000)
featurized_data = hashingTF.transform(apps_df)

# Now, create the IDF model from the raw features
idf = IDF(inputCol="raw_features", outputCol="tfidf_features")
idf_model = idf.fit(featurized_data)
apps_df = idf_model.transform(featurized_data)

# 3. Numerical features vector
print("3. Creating numerical features...")
numerical_cols = ["score", "ratings", "price", "minInstalls", "reviews",
                  "summary_length", "rating_quality", "popularity_score"]

# Assemble numerical features
assembler = VectorAssembler(
    inputCols=numerical_cols, outputCol="numerical_features")
apps_df = assembler.transform(apps_df)

print("=== Converting to Pandas for Advanced ML ===")
# Convert to Pandas for sklearn operations
pandas_df = apps_df.select("appId", "title", "genre", "summary", "developer",
                           "score", "free", "price", "ratings", "minInstalls", "reviews",
                           "rating_quality", "popularity_score", "price_category",
                           "count_features", "tfidf_features", "numerical_features").toPandas()

print(f"Pandas DataFrame shape: {pandas_df.shape}")

# Advanced Text Processing with sklearn
print("=== Advanced Text Processing with sklearn ===")

# Combine text fields for TF-IDF
pandas_df['combined_text'] = (
    pandas_df['summary'].fillna('') + ' ' +
    pandas_df['genre'].fillna('') + ' ' +
    pandas_df['developer'].fillna('')
).str.lower()

# Multiple TF-IDF approaches
print("Creating multiple TF-IDF matrices...")

# 1. Basic TF-IDF
tfidf_basic = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1, 1),
    min_df=2,
    max_df=0.95
)
tfidf_basic_matrix = tfidf_basic.fit_transform(pandas_df['combined_text'])

# 2. N-gram TF-IDF (1-2 grams)
tfidf_ngram = TfidfVectorizer(
    max_features=8000,
    stop_words='english',
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)
tfidf_ngram_matrix = tfidf_ngram.fit_transform(pandas_df['combined_text'])

# 3. Character-level TF-IDF
tfidf_char = TfidfVectorizer(
    max_features=3000,
    analyzer='char_wb',
    ngram_range=(3, 5),
    min_df=2
)
tfidf_char_matrix = tfidf_char.fit_transform(pandas_df['combined_text'])

print(f"TF-IDF Basic matrix shape: {tfidf_basic_matrix.shape}")
print(f"TF-IDF N-gram matrix shape: {tfidf_ngram_matrix.shape}")
print(f"TF-IDF Character matrix shape: {tfidf_char_matrix.shape}")

# Numerical features preprocessing
print("=== Preprocessing Numerical Features ===")
numerical_features = pandas_df[['score', 'ratings',
                                'price', 'minInstalls', 'reviews']].fillna(0)

# Standard scaling for numerical features
scaler = StandardScaler()
numerical_scaled = scaler.fit_transform(numerical_features)

# Combine features for hybrid approach

# Create dense numerical matrix
numerical_dense = csr_matrix(numerical_scaled)

# Combine TF-IDF with numerical features
combined_features_basic = hstack([tfidf_basic_matrix, numerical_dense])
combined_features_ngram = hstack([tfidf_ngram_matrix, numerical_dense])

print(f"Combined features (basic) shape: {combined_features_basic.shape}")
print(f"Combined features (n-gram) shape: {combined_features_ngram.shape}")

# Multiple Similarity Approaches
print("=== Computing Multiple Similarity Matrices ===")

# 1. Cosine similarity with basic TF-IDF
print("1. Computing cosine similarity (TF-IDF basic)...")
cosine_sim_tfidf = cosine_similarity(tfidf_basic_matrix)

# 2. Cosine similarity with n-gram TF-IDF
print("2. Computing cosine similarity (TF-IDF n-gram)...")
cosine_sim_ngram = cosine_similarity(tfidf_ngram_matrix)

# 3. Linear kernel similarity (faster for TF-IDF)
print("3. Computing linear kernel similarity...")
linear_sim = linear_kernel(tfidf_basic_matrix)

# 4. Combined features similarity
print("4. Computing combined features similarity...")
cosine_sim_combined = cosine_similarity(combined_features_basic)

print(f"Cosine similarity (TF-IDF) shape: {cosine_sim_tfidf.shape}")
print(f"Cosine similarity (N-gram) shape: {cosine_sim_ngram.shape}")
print(f"Linear kernel similarity shape: {linear_sim.shape}")
print(f"Combined similarity shape: {cosine_sim_combined.shape}")

# Advanced ML Models
print("=== Training Advanced ML Models ===")

# 1. Topic Modeling with LDA
print("1. Training LDA topic model...")
lda_model = LatentDirichletAllocation(
    n_components=20,
    random_state=42,
    max_iter=10,
    learning_method='online'
)
lda_topics = lda_model.fit_transform(tfidf_basic_matrix)

# 2. Non-negative Matrix Factorization
print("2. Training NMF model...")
nmf_model = NMF(
    n_components=50,
    random_state=42,
    max_iter=100
)
nmf_features = nmf_model.fit_transform(tfidf_basic_matrix)

# 3. K-Means Clustering
print("3. Training K-Means clustering...")
kmeans_model = KMeans(
    n_clusters=30,
    random_state=42,
    n_init=10
)
cluster_labels = kmeans_model.fit_predict(combined_features_basic.toarray())
pandas_df['cluster_label'] = cluster_labels

# 4. Random Forest for Rating Prediction (as additional features)
print("4. Training Random Forest for rating prediction...")
rf_features = pandas_df[['ratings', 'price',
                         'minInstalls', 'reviews']].fillna(0)
rf_target = pandas_df['score'].fillna(0)

# Only train if we have enough valid data
if len(rf_features) > 100 and rf_target.notna().sum() > 50:
    X_train, X_test, y_train, y_test = train_test_split(
        rf_features, rf_target, test_size=0.2, random_state=42
    )

    rf_model = RandomForestRegressor(
        n_estimators=50,
        random_state=42,
        max_depth=10
    )
    rf_model.fit(X_train, y_train)

    # Predict ratings for all apps
    predicted_ratings = rf_model.predict(rf_features)
    pandas_df['predicted_rating'] = predicted_ratings

    # Model evaluation
    y_pred = rf_model.predict(X_test)
    rf_mse = mean_squared_error(y_test, y_pred)
    rf_mae = mean_absolute_error(y_test, y_pred)
    print(f"Random Forest - MSE: {rf_mse:.4f}, MAE: {rf_mae:.4f}")
else:
    pandas_df['predicted_rating'] = pandas_df['score']
    rf_model = None

print("=== Advanced Feature Engineering Complete ===")
print(f"Final dataset shape: {pandas_df.shape}")
print(f"LDA topics shape: {lda_topics.shape}")
print(f"NMF features shape: {nmf_features.shape}")
print(f"Number of clusters: {len(set(cluster_labels))}")

# Advanced Recommendation Functions
print("=== Creating Advanced Recommendation Functions ===")


def get_recommendations_tfidf(title, similarity_matrix=None, method='tfidf_basic', num_recommendations=10):
    """
    Get app recommendations using various similarity approaches
    """
    try:
        if similarity_matrix is None:
            similarity_matrix = cosine_sim_tfidf

        # Find matching apps
        matching_apps = pandas_df[pandas_df['title'].str.contains(
            title, case=False, na=False)]

        if matching_apps.empty:
            print(f"No app found with title containing: {title}")
            return pandas_df.head(0)

        idx = matching_apps.index[0]

        # Get similarity scores
        sim_scores = list(enumerate(similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        # Exclude the app itself
        sim_scores = sim_scores[1:num_recommendations+1]

        # Get app indices
        app_indices = [i[0] for i in sim_scores]

        # Create recommendations with additional features
        recommendations = pandas_df.iloc[app_indices].copy()
        recommendations['similarity_score'] = [score[1]
                                               for score in sim_scores]
        recommendations['method'] = method

        return recommendations[['title', 'genre', 'developer', 'summary', 'score',
                                'free', 'price', 'ratings', 'similarity_score', 'method',
                                'rating_quality', 'popularity_score', 'cluster_label']]

    except Exception as e:
        print(f"Error getting TF-IDF recommendations: {e}")
        return pandas_df.head(0)


def get_recommendations_hybrid(title, num_recommendations=10):
    """
    Hybrid recommendation combining multiple approaches
    """
    try:
        # Get recommendations from different methods
        recs_tfidf = get_recommendations_tfidf(
            title, cosine_sim_tfidf, 'tfidf_basic', num_recommendations*2)
        recs_ngram = get_recommendations_tfidf(
            title, cosine_sim_ngram, 'tfidf_ngram', num_recommendations*2)
        recs_combined = get_recommendations_tfidf(
            title, cosine_sim_combined, 'combined_features', num_recommendations*2)

        if recs_tfidf.empty and recs_ngram.empty and recs_combined.empty:
            return pandas_df.head(0)

        # Combine and weight the recommendations
        all_recs = []

        # Weight different approaches
        weights = {'tfidf_basic': 0.3,
                   'tfidf_ngram': 0.4, 'combined_features': 0.3}

        for _, rec in recs_tfidf.iterrows():
            rec['weighted_score'] = rec['similarity_score'] * \
                weights['tfidf_basic']
            all_recs.append(rec)

        for _, rec in recs_ngram.iterrows():
            rec['weighted_score'] = rec['similarity_score'] * \
                weights['tfidf_ngram']
            all_recs.append(rec)

        for _, rec in recs_combined.iterrows():
            rec['weighted_score'] = rec['similarity_score'] * \
                weights['combined_features']
            all_recs.append(rec)

        # Create DataFrame and aggregate scores for same apps
        hybrid_df = pd.DataFrame(all_recs)

        if not hybrid_df.empty:
            # Group by title and aggregate scores
            aggregated = hybrid_df.groupby('title').agg({
                'weighted_score': 'mean',
                'similarity_score': 'mean',
                'genre': 'first',
                'developer': 'first',
                'summary': 'first',
                'score': 'first',
                'free': 'first',
                'price': 'first',
                'ratings': 'first',
                'rating_quality': 'first',
                'popularity_score': 'first'
            }).reset_index()

            # Sort by weighted score and return top recommendations
            final_recs = aggregated.sort_values(
                'weighted_score', ascending=False).head(num_recommendations)
            final_recs['method'] = 'hybrid'

            return final_recs[['title', 'genre', 'developer', 'summary', 'score',
                               'free', 'price', 'ratings', 'similarity_score', 'method',
                               'rating_quality', 'popularity_score']]

        return pandas_df.head(0)

    except Exception as e:
        print(f"Error getting hybrid recommendations: {e}")
        return pandas_df.head(0)


def get_recommendations_by_cluster(title, num_recommendations=10):
    """
    Get recommendations based on clustering
    """
    try:
        # Find the app and its cluster
        matching_apps = pandas_df[pandas_df['title'].str.contains(
            title, case=False, na=False)]

        if matching_apps.empty:
            return pandas_df.head(0)

        app_cluster = matching_apps.iloc[0]['cluster_label']

        # Get other apps in the same cluster
        cluster_apps = pandas_df[pandas_df['cluster_label'] == app_cluster]
        cluster_apps = cluster_apps[~cluster_apps['title'].str.contains(
            title, case=False, na=False)]

        # Sort by popularity score and rating quality
        recommendations = cluster_apps.sort_values(
            ['popularity_score', 'rating_quality'],
            ascending=[False, False]
        ).head(num_recommendations)

        return recommendations[['title', 'genre', 'developer', 'summary', 'score',
                                'free', 'price', 'ratings', 'rating_quality',
                                'popularity_score', 'cluster_label']]

    except Exception as e:
        print(f"Error getting cluster recommendations: {e}")
        return pandas_df.head(0)


def get_recommendations_by_topic(title, num_recommendations=10):
    """
    Get recommendations based on LDA topic modeling
    """
    try:
        matching_apps = pandas_df[pandas_df['title'].str.contains(
            title, case=False, na=False)]

        if matching_apps.empty:
            return pandas_df.head(0)

        app_idx = matching_apps.index[0]
        app_topics = lda_topics[app_idx]

        # Find most important topic for this app
        main_topic = np.argmax(app_topics)

        # Calculate topic similarity for all apps
        topic_similarities = []
        for i, topics in enumerate(lda_topics):
            # Use cosine similarity between topic distributions
            sim = np.dot(app_topics, topics) / \
                (np.linalg.norm(app_topics) * np.linalg.norm(topics))
            topic_similarities.append((i, sim))

        # Sort by topic similarity
        topic_similarities = sorted(
            topic_similarities, key=lambda x: x[1], reverse=True)
        # Exclude the app itself
        topic_similarities = topic_similarities[1:num_recommendations+1]

        # Get recommendations
        app_indices = [i[0] for i in topic_similarities]
        recommendations = pandas_df.iloc[app_indices].copy()
        recommendations['topic_similarity'] = [score[1]
                                               for score in topic_similarities]
        recommendations['main_topic'] = main_topic

        return recommendations[['title', 'genre', 'developer', 'summary', 'score',
                                'free', 'price', 'ratings', 'topic_similarity', 'main_topic']]

    except Exception as e:
        print(f"Error getting topic recommendations: {e}")
        return pandas_df.head(0)

# Keep existing functions but enhance them


def get_recommendations(title, cosine_sim=cosine_sim_tfidf, num_recommendations=10):
    """
    Enhanced version of original function using TF-IDF
    """
    return get_recommendations_tfidf(title, cosine_sim, 'enhanced_tfidf', num_recommendations)


# Enhanced popularity-based recommendations
def get_popular_apps_enhanced(num_recommendations=10, category=None):
    """
    Enhanced popular apps with multiple ranking criteria
    """
    try:
        df = pandas_df.copy()

        if category:
            df = df[df['genre'].str.contains(category, case=False, na=False)]

        # Multi-criteria popularity score
        df['enhanced_popularity'] = (
            df['score'] * 0.3 +
            df['rating_quality'] * 0.4 +
            df['popularity_score'] * 0.3
        )

        popular_apps = df.sort_values(
            'enhanced_popularity', ascending=False).head(num_recommendations)
        return popular_apps[['title', 'genre', 'developer', 'summary', 'score', 'free',
                             'price', 'ratings', 'enhanced_popularity', 'rating_quality']]
    except Exception as e:
        print(f"Error getting enhanced popular apps: {e}")
        return pandas_df.head(0)
    """
    Get app recommendations based on genre
    """
    try:
        genre_apps = pandas_df[pandas_df['genre'].str.contains(
            genre, case=False, na=False)]

        if genre_apps.empty:
            print(f"No apps found in genre: {genre}")
            return pandas_df.head(0)

        # Sort by score and return top apps
        recommendations = genre_apps.sort_values(
            'score', ascending=False).head(num_recommendations)
        return recommendations[['title', 'genre', 'developer', 'summary', 'score', 'free', 'price', 'ratings']]

    except Exception as e:
        print(f"Error getting genre recommendations: {e}")
        return pandas_df.head(0)


def get_popular_apps(num_recommendations=10):
    """
    Get most popular apps with enhanced scoring
    """
    return get_popular_apps_enhanced(num_recommendations)


def get_free_apps(num_recommendations=10):
    """
    Get top free apps based on enhanced criteria
    """
    try:
        free_apps = pandas_df[pandas_df['free'] == 1]
        free_apps_sorted = free_apps.sort_values(
            ['rating_quality', 'popularity_score'], ascending=[False, False])
        return free_apps_sorted[['title', 'genre', 'developer', 'summary', 'score',
                                 'free', 'price', 'ratings', 'rating_quality']].head(num_recommendations)
    except Exception as e:
        print(f"Error getting free apps: {e}")
        return pandas_df.head(0)


def get_paid_apps(num_recommendations=10):
    """
    Get top paid apps based on enhanced criteria
    """
    try:
        paid_apps = pandas_df[pandas_df['free'] == 0]
        paid_apps_sorted = paid_apps.sort_values(
            ['rating_quality', 'popularity_score'], ascending=[False, False])
        return paid_apps_sorted[['title', 'genre', 'developer', 'summary', 'score',
                                 'free', 'price', 'ratings', 'rating_quality']].head(num_recommendations)
    except Exception as e:
        print(f"Error getting paid apps: {e}")
        return pandas_df.head(0)


def get_apps_by_developer(developer_name, num_recommendations=10):
    """
    Get apps by specific developer with enhanced ranking
    """
    try:
        developer_apps = pandas_df[pandas_df['developer'].str.contains(
            developer_name, case=False, na=False)]
        if developer_apps.empty:
            print(f"No apps found from developer: {developer_name}")
            return pandas_df.head(0)

        developer_apps_sorted = developer_apps.sort_values(
            ['rating_quality', 'popularity_score'], ascending=[False, False])
        return developer_apps_sorted[['title', 'genre', 'developer', 'summary', 'score',
                                      'free', 'price', 'ratings', 'rating_quality']].head(num_recommendations)
    except Exception as e:
        print(f"Error getting developer apps: {e}")
        return pandas_df.head(0)


def get_app_statistics():
    """
    Enhanced dataset statistics
    """
    try:
        stats = {
            'total_apps': len(pandas_df),
            'free_apps': len(pandas_df[pandas_df['free'] == 1]),
            'paid_apps': len(pandas_df[pandas_df['free'] == 0]),
            'avg_score': pandas_df['score'].mean(),
            'avg_ratings': pandas_df['ratings'].mean(),
            'avg_rating_quality': pandas_df['rating_quality'].mean(),
            'avg_popularity_score': pandas_df['popularity_score'].mean(),
            'top_genres': pandas_df['genre'].value_counts().head(5).to_dict(),
            'score_range': f"{pandas_df['score'].min()} - {pandas_df['score'].max()}",
            'apps_with_zero_score': len(pandas_df[pandas_df['score'] == 0.0]),
            'num_clusters': len(pandas_df['cluster_label'].unique()),
            'price_categories': pandas_df['price_category'].value_counts().to_dict()
        }
        return stats
    except Exception as e:
        print(f"Error getting statistics: {e}")
        return {}


def export_recommendations_for_frontend(app_title, method='hybrid', num_recommendations=10):
    """
    Export recommendations with enhanced features
    """
    try:
        if method == 'hybrid':
            recommendations = get_recommendations_hybrid(
                app_title, num_recommendations)
        elif method == 'cluster':
            recommendations = get_recommendations_by_cluster(
                app_title, num_recommendations)
        elif method == 'topic':
            recommendations = get_recommendations_by_topic(
                app_title, num_recommendations)
        else:
            recommendations = get_recommendations_tfidf(
                app_title, cosine_sim_tfidf, method, num_recommendations)

        # Convert to dictionary format suitable for frontend
        frontend_data = {
            'query_app': app_title,
            'method': method,
            'recommendations': []
        }

        for _, row in recommendations.iterrows():
            app_data = {
                'title': row['title'],
                'genre': row['genre'],
                'developer': row['developer'],
                'summary': row['summary'],
                'score': float(row['score']) if pd.notna(row['score']) else 0.0,
                'free': bool(row['free']),
                'price': float(row['price']) if pd.notna(row['price']) else 0.0,
                'ratings': int(row['ratings']) if pd.notna(row['ratings']) else 0,
                'similarity_score': float(row.get('similarity_score', 0)) if pd.notna(row.get('similarity_score', 0)) else 0.0,
                'rating_quality': float(row.get('rating_quality', 0)) if pd.notna(row.get('rating_quality', 0)) else 0.0,
                'popularity_score': float(row.get('popularity_score', 0)) if pd.notna(row.get('popularity_score', 0)) else 0.0
            }
            frontend_data['recommendations'].append(app_data)

        return frontend_data

    except Exception as e:
        print(f"Error exporting recommendations for frontend: {e}")
        return {'query_app': app_title, 'method': method, 'recommendations': []}


def save_recommendations_json(app_title, method='hybrid', filename=None):
    """
    Save enhanced recommendations as JSON file
    """
    try:
        if filename is None:
            filename = f"recommendations_{app_title.replace(' ', '_')}_{method}.json"

        recommendations_data = export_recommendations_for_frontend(
            app_title, method)

        # Save to models directory
        os.makedirs("models", exist_ok=True)
        filepath = os.path.join("models", filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(recommendations_data, f, indent=2, ensure_ascii=False)

        print(f"Recommendations saved to: {filepath}")
        return filepath

    except Exception as e:
        print(f"Error saving recommendations JSON: {e}")
        return None
    
# Save the enhanced models and artifacts
print("\n=== Saving Enhanced Model Artifacts ===")
try:
    # Create directory if it doesn't exist
    os.makedirs("models", exist_ok=True)

    # Save the enhanced Pandas DataFrame
    pickle.dump(pandas_df, open("models/apps_list_enhanced.pkl", "wb"))

    # Save all similarity matrices
    pickle.dump(cosine_sim_tfidf, open("models/similarity_tfidf.pkl", "wb"))
    pickle.dump(cosine_sim_ngram, open("models/similarity_ngram.pkl", "wb"))
    pickle.dump(linear_sim, open("models/similarity_linear.pkl", "wb"))
    pickle.dump(cosine_sim_combined, open(
        "models/similarity_combined.pkl", "wb"))

    # Save the TF-IDF models
    pickle.dump(tfidf_basic, open("models/tfidf_basic_model.pkl", "wb"))
    pickle.dump(tfidf_ngram, open("models/tfidf_ngram_model.pkl", "wb"))
    pickle.dump(tfidf_char, open("models/tfidf_char_model.pkl", "wb"))

    # Save advanced ML models
    pickle.dump(lda_model, open("models/lda_model.pkl", "wb"))
    pickle.dump(nmf_model, open("models/nmf_model.pkl", "wb"))
    pickle.dump(kmeans_model, open("models/kmeans_model.pkl", "wb"))
    pickle.dump(scaler, open("models/scaler.pkl", "wb"))

    # Save LDA topics and NMF features
    np.save("models/lda_topics.npy", lda_topics)
    np.save("models/nmf_features.npy", nmf_features)

    # Save Random Forest model if trained
    if rf_model is not None:
        pickle.dump(rf_model, open("models/rf_rating_model.pkl", "wb"))

    # Save the Count Vectorizer vocabulary from Spark
    vocab = count_model.vocabulary
    pickle.dump(vocab, open("models/count_vectorizer_vocab.pkl", "wb"))

    # Save feature matrices (sparse matrices)
    from scipy.sparse import save_npz
    save_npz("models/tfidf_basic_matrix.npz", tfidf_basic_matrix)
    save_npz("models/tfidf_ngram_matrix.npz", tfidf_ngram_matrix)
    save_npz("models/combined_features_basic.npz", combined_features_basic)

    print("Enhanced model artifacts saved successfully!")
    print("=== Saved Files ===")
    print("- apps_list_enhanced.pkl: Enhanced app metadata and features")
    print("- similarity_*.pkl: Multiple similarity matrices")
    print("- tfidf_*.pkl: TF-IDF vectorizer models")
    print("- lda_model.pkl: LDA topic model")
    print("- nmf_model.pkl: NMF model")
    print("- kmeans_model.pkl: K-means clustering model")
    print("- scaler.pkl: Feature scaler")
    print("- *.npy: NumPy arrays for topics and features")
    print("- *.npz: Sparse feature matrices")
    if rf_model is not None:
        print("- rf_rating_model.pkl: Random Forest rating predictor")

except Exception as e:
    print(f"Error saving enhanced model artifacts: {e}")


print("\n=== Enhanced Recommendation System Setup Complete ===")
print("Multiple advanced models trained and ready for deployment!")
print("Available recommendation methods: TF-IDF, N-gram, Hybrid, Cluster, Topic-based")

# Stop Spark session
spark.stop()
