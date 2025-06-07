# Movie Recommendation System (aisc_wq25_movie_recommender)
>**_Beginner Project UC Davis AI Student Collective - Winter Quarter 2025_**

**TEAM:**
_Jeremy Elvander, Federico Aprile, Andy Tran, Chris Nguyen, Kavin Agarwal_

>⚠️ Note: This project was originally part of a shared group repository, which has since been removed by the owner. As a result, this repository was rebuilt independently from the ground up.

---

## Overview

This project is a personalized movie recommendation engine that:
- Stores, processes, and retrieves movie data using a PostgreSQL database
- Transforms raw movie metadata through feature engineering
- Analyzes user preferences and movie attributes to provide relevant recommendations

---

### Database Architecture

  Backend: PostgreSQL with SQLAlchemy ORM
  Schema:
  - Movies indexed by title (primary key)
  - Attributes include: actors, genres, writers, directors, Rotten Tomatoes score, release date
  - Optimized with arrays (for cast/crew) to reduce join complexity
  - Data stored in a custom schema (movie_info) for clarity and separation

### Data Collection & Processing (ETL)

  Extract:
  - Scrapes data from IMSDB and Rotten Tomatoes
  - Techniques: requests, lxml (XPath), and re (regex) for pattern extraction (e.g., writer names, years, URLs)

  Transform:
  - Handled missing/null values with defaults or estimates
  - Features like genres converted via Multi-Label Binarization
  - Numeric features (e.g. Rotten Tomatoes scores) normalized using MinMaxScaler

  Load:
  - Inserted into PostgreSQL using parameterized SQL and SQLAlchemy for secure, efficient transactions

### Feature Engineering & NLP

  Categorical Features:
  - Actors, writers, and directors embedded using Word2Vec to handle high-cardinality data
  - Genres converted to binary vectors via multi-label binarization

  Similarity Computation:
  - Movies compared using cosine similarity across embedded and engineered feature vectors
  - Full pairwise similarity matrix computed (cross-join style) to support recommendations

### Recommendation Methodology

  - Collect and structure movie data
  - Convert features to a uniform numerical representation
  - Compute similarity scores across the dataset
  - Store results for fast retrieval
  - Retrieve and rank recommendations based on content similarity

---

### Future Improvements

  - Add movie rating as an additional predictive feature
  - Expand dataset (currently ~1,000 movies)
  - Improve null/missing value handling strategies
  - Develop a public-facing frontend
  - Implement an API for recommendation access via HTTP requests

---

### Dependencies

Once within the cloned/downloaded directory, to install the necessary required packages simply run the following in the Terminal in an IDE (inside of a virtual environment ideally):

    pip install -r requirements.txt

---

### How to run it:

  Once within the cloned/downloaded directory, simply run this in the Terminal (on its own or in an IDE):

    python3 aisc_wq25_bp_model.py


