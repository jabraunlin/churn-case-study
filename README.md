# churn-case-study
Predicting churn from a ride share company's user data


pipeline_.py: file containing pipeline object with fit and transformation methods (for both data preparation and modeling).

implementation.ipynb: notebook containing data loading, cleaning, transformation, and results


## Overview

The goal of the case study was to create a model that would accurately classify customers that would churn, characterized by not calling a ride on the app for over 30 days. Data provided was aggregated for each user. Data included values for categories such as average driver rating, city, average surge the rider was charged, percent of trips taken on weekdays, number of trips taken in the first 30 days, days, date of last trip, average distance per trip, etc.

## EDA

I hypothesized that churn would have a strong relationship with the average surge pricing riders were being charged and average driver ratings. This relationship is graphed at the bottom of implementation.ipynb. Yellow dots, siginfying users who churned, were definitely being charged more on average than those who didn't. However, a strange relationship was observed where large amounts of users who churned were stacked at average user ratings of 1.0, 2.0, 3.0, 4.0, and 5.0. 

There was no data provided for how many rides each of these users took in total, but we can infer from the data with high confidence who was likely only a one-time user. To do this, I used the following condition. If a user had an average rating that was a whole number (i.e. 1.0, 2.0, 3.0, 4.0, and 5.0.), took one or less ride in the first 30 days, their weekday ride percentage was either 100% or 0%, and their average surge percent was a round number, they were identified as most likely a "One time user". A dummy column was created to identify these users. Churn rate was considerably higher among this subset, and the addition of this column significantly improved cross validation scores. 

Categorical values were dummified, and missing values were either dropped or imputed. 

## Model Fitting

The data was fitted to various models such as bagged decision trees, random forest, Adaboost, and SVM. Accuracy and recall were the main scoring metrics used. Recall was prioritized because it was assumed there would be a minimal negative impact on incorrectly identifying users who weren't planning on churning, and I wanted to minimize false negatives while maximizing true positives.

The random forest model performed best after testing on the validation set. Grid search was used to locate the best parameters. Grid searching helped reduce overfitting by lowering our tree count and tree depth. After testing the model on the test set, the model achieved a recall score of 93%, and an accuracy score of 86%.

## Further Analysis

Some obvious next steps to take would be to extract all users that were labeled "one time users" and to see if we could fit a separate model to both that subset and the other subset. Ensembling two models together might lead to better results. Furthermore, other models could be employed such as CatBoost and XGBoost. Fitting each of these models individually and grid searching over each of their parameters before comparing them could also positively impact our results. Unfortunately, this wasn't possible in the time allotted. 
