# Final Project for CSC 381: Recommender Systems Research

## Authors
- Joseph Brock '22
- Jake Carver '21
- Neil Patel '22
- Annabel Winters-McCabe '21

## Introduction
Hi! We are a group of Davidson College students researching Recommender Systems. The purpose of our work is to compare and contrast different recommender system algorithms, including user-based and item-based collaborative filtering, term frequency - inverse document frequency (TF-IDF), feature encoding (FE), matrix factorization methods with stochastic gradient descent (MF-SGD) and alternating-least-squares (MF-ALS), and a hybrid recommender algorithm (HBR) using accuracy and coverage as our metrics for comparison. Our experiments were primarily conducted using the MovieLens 100-K dataset, which holds "100000 movie ratings by 943 users on 1682 movies." [1]

## How to Use
1. Clone the GitHub Repo @ https://github.com/nepatel/rec-sys-final.git

2. Install any missing dependencies (copy, math, mathplotlib, numpy, openpyxl, os, pandas, pickle, sklearn) using pip3

3. Run 'recommendations.py' (NOTE: Directory path variable may need to be adjusted)

4. Read in a dataset with: R or RML

  => (R = critics, RML = MLK-100)

5. Choose your adventure :)

  - Content-Based Recommenders
    - Feature Encoding => (FE) => (CBR-FE)
    - Term Frequency - Inverse Document Frequncy => (TFIDF) => (CBR-TF)

  - Hybrid Recommender => (TFIDF) => (SIM) => (RD / RP / WD / WP) => (HBR)

  - Leave-One-Out-Cross-Validation => (FE / TFIDF / HBR)
    - NOTE: This function will also write results to an excel spreadsheet.

## References
[1] https://grouplens.org/datasets/movielens/100k/
