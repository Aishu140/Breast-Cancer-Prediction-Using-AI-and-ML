Breast Cancer Prediction Using Machine Learning
This project focuses on predicting breast cancer using machine learning techniques. The goal is to classify cancer types (malignant vs. benign) based on diagnostic data, enabling early detection and timely medical intervention. The project was developed in a Jupyter Notebook environment using Python and popular libraries such as Scikit-learn, Pandas, Matplotlib, and Seaborn.
Key Features
Data Preparation:
Loaded and cleaned the dataset, handling missing values and ensuring data quality.
Conducted exploratory data analysis (EDA) to understand the dataset's structure and distribution.
Feature Encoding:
Converted categorical data into numerical format using Label Encoding to make it suitable for machine learning models.
Data Visualization:
Used Matplotlib and Seaborn to visualize data distributions, class imbalances, and relationships between features.
Feature Engineering:
Split the dataset into training and testing sets.
Applied feature scaling using StandardScaler to normalize the data and improve model performance.
Model Building:
Implemented and evaluated multiple machine learning models, including:
Logistic Regression
Decision Tree
Random Forest
Support Vector Machine (SVM)
K-Nearest Neighbors (KNN)
Used a custom function to train, evaluate, and compare model performance based on accuracy, precision, recall, and F1-score.
Hyperparameter Tuning:
Optimized the Random Forest model using GridSearchCV to find the best hyperparameters, such as the number of trees and maximum depth.
Model Deployment:
Saved the best-performing model using the pickle library for future use and deployment in real-world applications.
Results
The Random Forest model achieved the highest accuracy of 96.49%, making it the best-performing model for this dataset.
Detailed classification reports were generated for each model, providing insights into precision, recall, and F1-scores for both malignant and benign classes.
Hyperparameter tuning further improved the Random Forest model's performance, demonstrating the importance of optimization in machine learning pipelines.
