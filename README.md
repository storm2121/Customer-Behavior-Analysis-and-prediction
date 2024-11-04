# Customer-Behavior-Analysis-and-prediction

Overview

This project is a comprehensive data analysis and predictive modeling tool developed to assist CIH Bank in understanding customer transaction behaviors and improving service offerings. It leverages advanced machine learning techniques to provide insights and predictive capabilities.

Key Features

Data Integration: Combines client, transaction, and operation datasets to create a unified analysis base.

Preprocessing: Includes data cleaning, handling of missing values, and feature scaling using StandardScaler.

Clustering Analysis: Utilizes KMeans clustering for customer segmentation and profiling.

Predictive Modeling: Employs a Random Forest Classifier for transaction type prediction.

Interactive GUI: Built with Tkinter for user data input and real-time predictions.

Data Visualization: Generates comprehensive plots such as age group distributions, average transaction amounts, and transaction frequency patterns.

Methods Used

Data Preprocessing: Imputation of missing data, normalization with StandardScaler, and data balancing with SMOTE.

Clustering: KMeans algorithm for customer segmentation.

Classification Model: Random Forest Classifier for predicting transaction types.

Dimensionality Reduction: PCA for visual representation of clusters.

Visualization: Utilized matplotlib and seaborn for visual insights.

Data Insights

Age Group Analysis: Visualizes transaction frequency and average transaction amounts by age groups.

Transaction Patterns: Shows days since the last transaction to understand customer activity.

Customer Segmentation: Clustering analysis highlights key customer profiles and behavior patterns for targeted strategies.

Results

Model Performance: Achieved high accuracy with the Random Forest Classifier.

Feature Importance: Identifies which features contribute most to predictive power.

Cluster Analysis: Highlights different customer profiles for tailored service approaches.

Dependencies

pandas

numpy

matplotlib

seaborn

scikit-learn

imbalanced-learn

tkinter
