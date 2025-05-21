# Weighted Season Regression: Predicting Premier League Outcomes
This project predicts final league standings and total points for Premier League teams using a weighted regression model based on past seasons. Each team's historical performance (from 2006 to 2017) is processed into features, with more recent seasons given higher importance using manually assigned weights. The goal is to forecast the 2017–2018 season outcome.

The model allows for:

Prediction of final league positions and points based on historical season data

Weighted feature engineering: recent seasons have more influence on predictions

Use of either Linear Regression or Random Forest Regression

Evaluation via cross-validation (with MAE and RMSE)

Export of results to .csv and visual comparison with real outcomes

A key element of this approach is the use of a custom weighting strategy to reflect changing team dynamics over time — instead of treating all past seasons equally, recent seasons carry more predictive power. This helps account for transfers, managerial changes, and form trends.

Additionally, a short presentation (in English) is available and explains the concept behind the model design, the weighting scheme, and the results — it's recommended if you're curious about the regression pipeline and prediction logic!
