# Credit card fraud detection
# Aim
To detect fraudulent credit card transactions using different Machine Learning models by training them to learn patterns behind fraudulent behaviour.

# Dataset
Contains the link to the “Credit Card Fraud Detection” dataset that is publicly available on Kaggle.
#### Description of the dataset:
The dataset contains transactions made by credit cards in September 2013 by European cardholders.
This dataset presents transactions that occurred in two days, where there is 492 frauds out of 284,807 transactions.  

It contains only numerical input variables which are the result of a PCA transformation of confidential transactions and locations of transactions. 
Features V1, V2, … V28 are the principal components obtained with PCA.
The features which have not been transformed with PCA are 'Time', 'Class' and 'Amount'. 
Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. 
Feature 'Amount' is the transaction Amount. 
Feature 'Class' is the target variable and it takes value 1 in case of fraud and 0 otherwise.

# Data Visualization
Visualisation of the dataset was done by using histogram distribution and all the features were correlated using heatmap.

# Models
1. Random Forest:
   A random forest is a supervised machine learning algorithm that can be used for both classification and regression tasks. The model works by sampling the training dataset, building multiple decision trees, and then having the output of the decision trees determine a prediction.

2. CatBoost:
   The CatBoost algorithms works by building decision trees consecutively and minimizing loss with each new decision tree that is built. It is designed to work well with imbalanced data, which makes the algorithm perfect for use in fraud detection.

3. Isolation forest:
   Isolation Forest is an unsupervised learning method, meaning that it does not require any truth-marking to make predictions, and only learns from patterns it finds in the training data. It is a tree-based algorithm used for anomaly detection. The algorithm works by using decision trees to isolate outliers from the data.

4. Logistic Regression:
   Logistic regression is a fundamental classification technique. It is an algorithm that measures the probability of a binary response as the value of response variable based on the mathematical equation relating it with the predictor variables.
   
5. XGBoost Method:
   XGBoost is a powerful approach for building supervised regression models. It is based on the Gradient Boosting model which uses the boosting technique of ensemble learning where the underfitted data of the weak learners are passed on to the strong learners to increase the strength and accuracy of the model.

6. KNN Classifier:
   K-nearest neighbor classifier is one of the introductory supervised classifiers. K-nearest neighbor classifier algorithms predict the target label by finding the nearest neighbor class. The closest class is identified using the distance measures like Euclidean distance.
   
7. GaussianNB:
   Gaussian Naive Bayes is a variant of Naive Bayes that follows Gaussian normal distribution and supports continuous data. Naive Bayes are a group of supervised machine learning classification algorithms based on the Bayes theorem. It is a simple classification technique, but has high functionality.
  
# Conclusion
The dataset was highly unbalanced, the positive class (frauds) account for 0.172% of all transactions. To resolve class imbalance, we used Synthetic Minority Over-sampling Technique (SMOTE) and applied XGBoost on the oversampled dataset. Also used Catboost model to resolve imbalance in dataset.
For measuring the accuracy score of each of the algorithms, we calculated and plotted area under ROC curve. 
#### ROC AOC of all the models:
1. Isolation forest: 0.946
2. Catboost: 0.973
3. XG Boost: 0.99
4. Logistic Regression: 0.771
5. Gaussian Naive Bayes: 0.8919
6. KNN Classifier: 0.50775

Concluding, XGBoost model including SMOTE applied on the model as the efficient model.




