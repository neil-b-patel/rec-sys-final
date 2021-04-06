# Final Project for CSC 381: Recommender Systems Research

## Authors
- Joseph Brock '22
- Jake Carver '21
- Neil Patel '22
- Annabel Winters-McCabe '21

## Introduction
Hi! We are a group of Davidson College students researching Recommender Systems.

## How to Use
1. Clone the GitHub Repo @ https://github.com/nepatel/rec-sys-final.git

2. Run 'recommendations.py' (NOTE: Directory path variable may need to be adjusted)

3. Read in a dataset with: R or RML 

  => (R = critics, RML = MLK-100)

4. Generate a similarity matrix with: Sim or Simu 

  => (Sim = Item-Based, Simu = User-Based)

5. Select a signicance similarity weighting: 0, 25, 50
  
  => (0 = NONE, 25 = n/25, 50 = n/50); where n is the number of shared items between two users

6. Select a minimum similarity threshold: >0, >0.3, >0.5
   
  => (0.3 or 3 = >0.3, 0.5 or 5 = >0.5, otherwise default to >0)

7. Select a subcommand: RD, RP, WD, WP 

  => (R = Read, W = Write, D = Euclidean Distance, P = Pearson Correlation)

8. Test metrics of accuracy with: LCVSIM (NOTE: LCV is deprecated)

## References
[1] https://grouplens.org/datasets/movielens/100k/
