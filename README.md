# Predicting and Understanding Song Popularity on Spotify

## Project Overview

This project aims to analyze and predict song popularity on Spotify using audio features and genre information. The goal is not only to build predictive models, but also to understand which musical characteristics are most associated with popularity.

This project follows a complete data science pipeline, including data collection, cleaning, feature extraction, visualization, and modeling.

---

## Project Goals

### Goal 1: Identify significant features associated with popularity
Analyze the relationship between audio features (e.g., danceability, energy, loudness, valence) and popularity using correlation analysis and visualization techniques.

### Goal 2: Predict song popularity (Regression)
Train regression models (Linear Regression, Random Forest) to predict popularity scores (0–100), evaluated using RMSE and R².

### Goal 3: Classify high-popularity songs (Classification)
Convert popularity into a binary classification task (e.g., popularity ≥ 70) and evaluate using accuracy and F1-score.

### Goal 4: Compare models and interpret results
Compare linear and nonlinear models, and analyze feature importance to understand which features contribute most to prediction.

---

## Data Collection

### Dataset

Spotify Tracks Dataset (Kaggle):  
https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset

The dataset contains over 100,000 songs with popularity scores and numerical audio features such as danceability, energy, loudness, speechiness, acousticness, instrumentalness, valence, tempo, and duration. It also includes genre information.

### Data Collection Method

The dataset is downloaded from Kaggle and loaded into Python using pandas. A data ingestion script is used to load and preprocess the dataset to ensure reproducibility.

---

## Data Cleaning

- Removed unnecessary index column (`Unnamed: 0`)
- Dropped rows with missing values
- Verified data types for all features
- Checked for duplicates and inconsistencies

---

## Feature Engineering

In addition to raw audio features, new features will be created to improve model performance:

- Log-transformed duration (`log_duration`)
- Interaction features (e.g., energy × loudness)
- Genre encoding (categorical to numerical)
- Popularity buckets (for classification)

---

## Exploratory Data Analysis

The following visualizations are used to understand the dataset:

- Popularity distribution
- Correlation heatmap
- Feature vs. popularity scatter plots
- Genre-based comparisons

---

## Modeling

### Regression Models
- Linear Regression (baseline)
- Random Forest Regressor (nonlinear model)

### Classification Models
- Logistic Regression
- Random Forest Classifier

### Evaluation Metrics
- Regression: RMSE, R²  
- Classification: Accuracy, F1-score  

---

## Results and Analysis

The project compares model performance and highlights:

- Nonlinear models outperform linear models  
- Certain features (e.g., energy, loudness, danceability) are more predictive  
- Popularity is influenced by multiple interacting factors  

---

## Visualization

- Feature importance plots  
- Actual vs predicted values  
- Residual analysis  
- Genre-based performance comparison  

---

## Project Timeline

### Week 1–2: Data Collection & Setup
- Download dataset and set up environment  
- Implement data loading pipeline  

### Week 3: Data Cleaning
- Handle missing values and remove unnecessary columns  
- Validate dataset  

### Week 4–5: EDA & Visualization
- Generate plots and analyze relationships  

### Week 6: Baseline Modeling
- Train Linear Regression and Logistic Regression  

### Week 7: Advanced Modeling
- Train Random Forest models  
- Compare performance  

### Week 8: Final Analysis & Report
- Interpret results  
- Generate visualizations  
- Finalize report  

---

## Reproducibility

- Code organized into notebooks and scripts  
- Dependencies listed in `requirements.txt`  
- All steps documented in this README  

---

## Project Structure