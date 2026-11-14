# Collaborative Filtering for Data Imputation

## Dataset

Using the Student Performance dataset:

Cortez, P. (2008). Student Performance [Dataset](https://archive.ics.uci.edu/dataset/320/student-performance). UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T.

## predict_g3
Impute g3 from the other features.

In CF terms, the student features are the users and the test scores for g3 are the items

Scenario 1:
user = student (every feature including G1 & G2)
rating = G3

Scenario 2:
user = student (every feature excluding G1 & G2)
rating = G3


## random masking
Mask a set% of the cells at random and compare different imputation techniques.

