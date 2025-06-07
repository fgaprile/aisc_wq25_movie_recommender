import psycopg2
from psycopg2.extensions import AsIs
from sqlalchemy import create_engine, text as sql_text

import requests
import lxml.html as lx
import re
import time
import nltk
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

from sklearn.preprocessing import normalize
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os
load_dotenv()  # Load from .env

from gensim.models import Word2Vec

db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_host = os.getenv("DB_HOST")
db_port = os.getenv("DB_PORT")
db_name = os.getenv("DB_NAME")


def db_data_retrieval():
    # Create a SQLAlchemy engine connected to PostgreSQL database
    db_eng = create_engine(
        f'postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}',
        connect_args={'options': '-c search_path=public'},
        isolation_level='READ COMMITTED'
    )


    
    # Uncomment below for debugging SQL queries and connection pool
    #    , echo=True)
    #    , echo_pool="debug")

    print("Successfully created db engine.")

    # Define SQL query to retrieve all movie records from movie_info.movies table
    sql = """
        SELECT * FROM movie_info.movies;
        """
    # Execute the query and return results as a pandas DataFrame
    return pd.read_sql_query(sql, db_eng)


def feature_engineering(df):
    # Data cleaning and preparation before modeling

    # For 'actors' and 'writers' columns, replace missing or invalid entries with empty lists
    df['actors'] = df['actors'].apply(lambda x: x if isinstance(x, list) else [])
    df['writers'] = df['writers'].apply(lambda x: x if isinstance(x, list) else [])
    # For 'director', replace missing values with empty string
    df['director'] = df['director'].fillna('')

    # For numerical columns:
    # - Replace missing 'rt_score' with column mean
    df['rt_score'] = df['rt_score'].fillna(df['rt_score'].mean())
    # - Convert 'date' to numeric, coercing errors to NaN
    df['date'] = pd.to_numeric(df['date'], errors='coerce')
    # - Replace missing 'date' values with rounded mean of date
    df['date'] = df['date'].fillna(df['date'].mean().round().astype(int))

    # Encode genres as a binary indicator matrix (one-hot encoding for multilabel)
    mlb = MultiLabelBinarizer()
    genres_encoded = pd.DataFrame(
        mlb.fit_transform(df['genres']),
        columns=mlb.classes_,
        index=df.index
    )

    # Prepare a list of lists combining actors, writers, and director names for Word2Vec training
    all_names = []
    for _, row in df.iterrows():
        names = []
        # Flatten actors list if nested, convert all to strings
        for actor in row['actors'] if isinstance(row['actors'], list) else [row['actors']]:
            if isinstance(actor, list):
                names.extend(str(sub_actor) for sub_actor in actor)
            else:
                names.append(str(actor))

        # Similarly flatten and convert writers
        for writer in row['writers'] if isinstance(row['writers'], list) else [row['writers']]:
            if isinstance(writer, list):
                names.extend(str(sub_writer) for sub_writer in writer)
            else:
                names.append(str(writer))

        # Add director name as string
        director = row['director']
        names.append(str(director))

        # Append this movie's combined names to all_names
        all_names.append(names)

    # Train a Word2Vec model on the combined names with specified parameters
    model = Word2Vec(all_names, vector_size=100, window=5, min_count=1, workers=4)

    # Define a helper function to get the average word embedding vector for a list of names
    def get_average_embedding(names):
        embeddings = []
        for name in names:
            # Only add embedding if name exists in the Word2Vec vocabulary
            if name in model.wv:
                embeddings.append(model.wv[name])
        if embeddings:
            # Average all embeddings
            return sum(embeddings) / len(embeddings)
        else:
            # If no embeddings found, return zero vector of model's vector size
            return [0] * model.vector_size

    # Compute average embedding vectors for actors, writers, and directors per movie
    df['actor_embedding'] = df['actors'].apply(get_average_embedding)
    df['writer_embedding'] = df['writers'].apply(get_average_embedding)
    df['director_embedding'] = df['director'].apply(
        lambda x: get_average_embedding(x if isinstance(x, list) else [str(x)])
    )

    # Scale numerical features rt_score and date to range [0, 1]
    scaler = MinMaxScaler()
    df[['rt_score_scaled', 'date_scaled']] = scaler.fit_transform(df[['rt_score', 'date']])

    # Convert embedding columns from lists to numpy arrays for easier manipulation
    actor_embeddings = np.array(df['actor_embedding'].tolist())
    writer_embeddings = np.array(df['writer_embedding'].tolist())
    director_embeddings = np.array(df['director_embedding'].tolist())

    # Combine all feature vectors into a single feature matrix:
    # concatenating genre encoding + actor embeddings + writer embeddings + director embeddings + scaled numerical features
    feature_matrix = np.hstack([
        genres_encoded.values,
        actor_embeddings,
        writer_embeddings,
        director_embeddings,
        df[['rt_score_scaled', 'date_scaled']].values
    ])

    return feature_matrix


def similarity_scores(feature_matrix):
    # Compute cosine similarity matrix between all movies based on their feature vectors
    similarity_matrix = cosine_similarity(feature_matrix)
    return similarity_matrix


def recommend_movies(movie_title, df, similarity_matrix, top_n=5):
    # Find index of the input movie title in the dataframe
    idx = df.index[df['title'] == movie_title].tolist()[0]

    # Retrieve similarity scores for the input movie against all others
    sim_scores = list(enumerate(similarity_matrix[idx]))

    # Sort similarity scores in descending order, most similar first
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Select indices of top_n most similar movies excluding the input movie itself (which will have similarity 1)
    top_indices = [i[0] for i in sim_scores[1:top_n + 1]]

    # Return the titles of these recommended movies
    return df.iloc[top_indices]['title'].tolist()


# Run the entire recommendation pipeline

# Step 1: Retrieve movie data from the database
df = db_data_retrieval()
print(df['title'])  # Print all movie titles

# Step 2: Perform feature engineering on the dataframe to create feature matrix
transformed_features = feature_engineering(df)

# Step 3: Calculate similarity matrix based on feature vectors
simMatrix = similarity_scores(transformed_features)

# Step 4: Get movie recommendations for a given movie title
recommendations = recommend_movies('Superbad', df, simMatrix)
print(recommendations)