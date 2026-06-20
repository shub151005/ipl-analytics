# IPL Analytics Dashboard

> Data Science meets Cricket Intelligence — 17 seasons, 1095 matches, statistically validated insights.

**Live Demo:** https://your-app-name.streamlit.app

---

## What This Project Does

A full statistical analysis and machine learning platform built on the complete IPL dataset (2008-2024). Combines exploratory data analysis, hypothesis testing, dimensionality reduction, and a live win probability prediction model.

---

## Key Features

### Exploratory Data Analysis
- Season-wise match trends
- Team performance rankings
- Player batting and bowling statistics
- Match phase analysis (Powerplay/Middle/Death overs)

### Statistical Hypothesis Testing
- **Chi-Square Test** — Does toss decision affect match outcome?
- **Independent T-Test** — Powerplay vs Death over scoring rates
- **One-way ANOVA** — Do venues have significantly different scoring patterns?
- **Shapiro-Wilk Test** — Distribution normality testing on innings scores
- **Binomial Test** — Batting first vs chasing win rate significance

### Dimensionality Reduction
- PCA on batting statistics (76% variance explained in 2D)
- K-Means clustering of player batting styles
- Component loading interpretation

### Machine Learning
- Win probability predictor using Gradient Boosting (AUC 0.869)
- Real-time match state input → live win probability
- Feature importance analysis

---

## Key Findings

| Finding | Statistical Evidence |
|---|---|
| Fielding first significantly increases win probability | Chi-square p=0.0104 |
| Death overs score significantly faster than Powerplay | T-test p<0.0001 |
| Chasing teams win more often (54.1% vs 45.9%) | Binomial test p=0.007 |
| Venue scoring reputation is statistically a myth | ANOVA p=0.39 (not significant) |
| Innings scores are NOT normally distributed (negative skew) | Shapiro-Wilk p<0.0001 |

---

## Tech Stack
Python · Pandas · NumPy · Scikit-learn · SciPyPlotly · Streamlit

---

## Local Setup

```bash
git clone https://github.com/shub151005/ipl-analytics.git
cd ipl-analytics
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m streamlit run 08_streamlit_app.py
```

---

## Dataset

IPL Complete Dataset (2008-2024) — Kaggle
https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020

---

## Project Structure
ipl-analytics/

├── 01_data_exploration.ipynb

├── 02_data_cleaning.ipynb

├── 03_eda_part1.ipynb

├── 04_eda_part2.ipynb

├── 05_hypothesis_testing.ipynb

├── 06_pca_analysis.ipynb

├── 07_win_probability.ipynb

├── 08_streamlit_app.py

├── matches_clean.csv

├── deliveries_clean.csv

└── requirements.txt

---

## Author

**Subham Mazumdar**
CSE Student — Assam Down Town University
Built as part of a Data Science portfolio focused on applied statistics and GATE DA preparation.