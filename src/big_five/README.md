# Collaborative Filtering for Data Imputation: Big Five

## Dataset

Big Five Personality Test (IPIP Big-Five Factor Markers), Open Psychometrics.
A 3,000-respondent sample distributed with R Journal article RJ-2025-021:
https://raw.githubusercontent.com/rjournal/rjournal.github.io/master/_articles/RJ-2025-021/data/BFPT.csv

Full dataset (about 1M respondents): https://www.kaggle.com/datasets/tunguz/big-five-personality-test

- 50 Likert questions (1-5), 10 per trait: EXT, EST, AGR, CSN, OPN
- 0 means "question skipped", so it is treated as missing
- Rows that are fully empty (6) or fully skipped (3) are dropped, leaving 2,991 respondents
- 974 genuinely missing cells

In CF terms, respondents are the users and questions are the items.

## Code

Reuses `random_masking.py` and `predict_g3.py` from `src/student_performance` unchanged. Only the data loading differs.

```
python run_bigfive.py BFPT.csv masking   # random masking (3% of cells, 30 repeats)
python run_bigfive.py BFPT.csv predict   # predict each question from the others
```

`predict_target.py` generalises `predict_g3` to a survey with no single target column.
- Every question takes a turn as the target.
- Scenario 2 drops the 2 questions most correlated with the target, chosen on the training rows. This is the analogue of dropping G1 and G2.

## Results

Full output: `results_masking.txt`, `results_predict.txt`.

Random masking (nRMSE, 1.0 = column mean):

| Method | nRMSE |
| --- | --- |
| column mean | 1.000 |
| item-CF (PCC) | 0.819 |
| user-CF (kNN) | 0.849 |
| MICE | 0.772 |

Predicting each question (MAE, mean over 50 questions):

| Method | All other questions | Without top-2 correlated |
| --- | --- | --- |
| item-CF (PCC) | 0.722 | 0.788 |
| user-CF (cosine) | 0.816 | 0.860 |
| global mean | 0.972 | 0.972 |

item-CF beats the global mean on all 50 questions in scenario 1, and on 49/50 in scenario 2.
