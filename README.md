Credit Card Fraud Detection
A machine learning project for detecting fraudulent credit card transactions using supervised classification on an imbalanced dataset.
Overview
This GROUP project as part of the Rice Machine Learning Club builds a fraud detection pipeline on a dataset of ~42,800 transactions (99% non-fraudulent, 1% fraudulent). It covers end-to-end data preprocessing, feature engineering, exploratory data analysis, and model training.
Dataset

Source: fraudDataset.csv
Size: 42,831 transactions × 23 features
Target: is_fraud (0 = legitimate, 1 = fraudulent)
Class imbalance: ~397 fraud cases out of 42,831 total (~0.93%)

Features include: transaction timestamp, merchant, category, amount, cardholder demographics, location coordinates, and job.
Project Structure
├── fraud_detection.ipynb   # Main notebook
├── fraudDataset.csv        # Raw dataset (not tracked)
└── README.md
Pipeline
1. Data Cleaning

Dropped null rows (7 rows affected)
Removed irrelevant columns: Unnamed: 0, cc_num, trans_num, first, last

2. Feature Engineering

Age derived from dob
Unix timestamp converted to datetime
Target encoding for job, merchant, and zip (encoded by mean fraud rate per group)
One-hot encoding for gender and category
Dropped raw location fields: lat, long, street, city

3. Exploratory Data Analysis

Class distribution (fraud vs. non-fraud)
Transaction amount distribution by fraud status
Fraud rate by merchant category
Fraud rate by state (top 10)
City population distribution by fraud status
Correlation heatmap for numerical features

4. Modeling

80/20 stratified train/test split
Models: (in progress)

Requirements
pandas
numpy
scikit-learn
matplotlib
seaborn
Install with:
bashpip install pandas numpy scikit-learn matplotlib seaborn
Usage

Place fraudDataset.csv in the project root
Open fraud_detection.ipynb in Jupyter or Google Colab
Run all cells top to bottom

Results
MetricScoreModelTBDAccuracyTBDFraud RecallTBDF1 ScoreTBD
Notes

Dataset is heavily imbalanced — accuracy alone is not a reliable metric; recall on the fraud class is prioritized.
Target encoding was applied to high-cardinality categorical features (job, merchant, zip) to avoid dimensionality explosion.
