# CS506-Project - Predicting Song Popularity Using Spotify Audio Features

Project Description:
This project aims to analyze the relationship between Spotify audio features and song popularity, and to build predictive models that estimate a song’s popularity score based on these features.
Spotify provides numerical audio descriptors such as danceability, energy, tempo, loudness, valence, acousticness, instrumentalness, and speechiness. These features capture measurable aspects of music and allow for quantitative analysis. The primary objective of this project is to determine which features are most strongly associated with popularity and whether they can be used to accurately predict popularity scores.

Project Timeline:
Week 1–2: Data Collection & Setup
•	Register and configure Spotify API access
•	Develop scripts to collect song audio features and popularity data
•	Store and structure dataset

Week 3: Data Cleaning & Preprocessing
•	Handle missing values and duplicates
•	Normalize/standardize features
•	Perform initial data quality checks

Week 4–5: Exploratory Data Analysis (EDA)
•	Analyze feature distributions
•	Compute correlations between features and popularity
•	Generate visualizations (heatmaps, scatter plots, histograms)

Week 6: Baseline Modeling
•	Implement Linear Regression / Ridge Regression
•	Evaluate using RMSE and R²
•	Perform cross-validation
Week 7: Advanced Modeling & Comparison
•	Implement Random Forest or Gradient Boosting
•	Compare performance with baseline models
•	Analyze feature importance

Week 8: Final Analysis & Report Preparation
•	Interpret results
•	Discuss limitations and potential improvements
•	Prepare final report and visualizations
 
Project Goals:

Goal 1: Identify significant audio features associated with popularity.
Examine the relationship between Spotify audio features (danceability, energy, tempo, valence, loudness, acousticness, instrumentalness, speechiness) and popularity scores using correlation analysis and feature importance methods.

Goal 2: Build a predictive regression model for song popularity.
Train at least two regression models (e.g., Linear Regression and Random Forest) to predict popularity scores (0–100). Model performance will be evaluated using RMSE and R². The goal is to achieve a model performance that improves upon a baseline mean predictor.

Goal 3: Compare model performance and interpret results.
Compare different models using cross-validation and determine which features contribute most to prediction performance using feature importance analysis.

Goal 4: Classify songs as “high popularity” vs “low popularity.”
Convert popularity into a binary classification task (e.g., popularity ≥ 70) and evaluate performance using accuracy and F1-score.
 
Data Collection Plan:
Data Source:
The dataset used for this project is the Spotify Tracks Dataset available on Kaggle:

https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset

This dataset contains over 100,000 tracks along with their popularity scores and a comprehensive set of numerical audio features, including danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo, and track duration. These variables quantitatively describe musical characteristics and will serve as predictor features in the modeling process. The dataset also includes genre information, which may be used for exploratory analysis.

Data Collection Method:
The dataset will be downloaded directly from Kaggle in CSV format and imported into Python using the pandas library. Data preprocessing will include removing duplicate entries, handling missing values, verifying data types, and standardizing numerical features where appropriate. The cleaned dataset will then be split into training and testing sets for model development and evaluation.
