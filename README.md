# Credit-Card-Fraud-Detection

This project focuses on detecting fraudulent credit card transactions using a highly imbalanced dataset, with only 0.17% of the transactions labeled as fraudulent. It utilizes a variety of preprocessing techniques and machine learning models to maximize detection accuracy while minimizing false positives.

Dataset
The dataset contains 284,807 transactions with 30 anonymized features (V1 to V28), along with Time, Amount, and a binary Class label indicating fraud (1) or legitimate (0).
Available on Kaggle: Credit Card Fraud Detection Dataset.
Key Steps
Data Preprocessing:

Scaled the Time and Amount columns using RobustScaler.
Handled class imbalance through undersampling and oversampling techniques (SMOTE).
Removed duplicates and checked for missing values.
Exploratory Data Analysis:

Visualized transaction distributions and class imbalance.
Analyzed correlations between features using heatmaps.
Modeling:

Tested multiple machine learning algorithms:
Logistic Regression
Random Forest Classifier
Decision Tree Classifier
XGBoost Classifier
Evaluated models using metrics like precision, recall, F1-score, and accuracy.
Results:

Achieved over 99% accuracy using Random Forest and XGBoost on oversampled data.
Balanced precision and recall to minimize false positives.
Code Example
Data Loading and Exploration
python
Copy
Edit
import pandas as pd

data = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
print(data.head())
Model Evaluation
python
Copy
Edit
from sklearn.metrics import classification_report

for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"{name}:\n{classification_report(y_test, y_pred)}")
Deployment
Saved the final model as a .pkl file using joblib for deployment.
Example usage:
python
Copy
Edit
import joblib

model = joblib.load('Credit_card_model.pkl')
pred = model.predict(new_data_point)
print("Fraud" if pred[0] == 1 else "No Fraud")
Technologies Used
Python
Pandas, NumPy
Scikit-learn
XGBoost
Seaborn, Matplotlib
Results Visualization

Future Improvements
Explore deep learning models for enhanced fraud detection.
Implement real-time fraud detection using streaming frameworks.
Author
Monirul Islam
LinkedIn | GitHub | md08monirul@gmail.com
