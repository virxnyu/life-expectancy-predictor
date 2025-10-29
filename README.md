# Life Expectancy Prediction using Machine Learning
# 1. Project Overview

This project aims to predict the life expectancy of a country based on key socio-economic and health indicators using various machine learning algorithms.
By analyzing global health and economic data, this model helps identify the factors most strongly influencing life expectancy — providing valuable insights for policymakers, researchers, and healthcare planners.

# 2. Why This Dataset Was Chosen

The Life Expectancy Dataset from the World Health Organization (WHO) was chosen because it contains a rich combination of medical, demographic, and economic indicators that directly influence public health outcomes.

Key Advantages

Comprehensive features (GDP, schooling, alcohol consumption, disease prevalence, etc.)

Global coverage across multiple countries and years

Real, interpretable data with direct social impact

Unlike typical datasets, it allows exploration of multi-dimensional relationships between lifestyle, economy, healthcare, and longevity, making it ideal for building regression models that reflect real-world complexity.

# 3. Objective

To develop and compare multiple regression models capable of accurately predicting Life Expectancy given socio-economic and health parameters.
The project also identifies the most influential factors contributing to longevity using feature importance analysis.

# 4. Data Preprocessing and Preparation
Data Cleaning

Removed missing target values (Life expectancy).

Handled missing numerical values using median imputation (robust against outliers).

Removed the Country column (too many unique values).

Encoded the Status column (Developed / Developing) using one-hot encoding.

Feature Scaling

Applied StandardScaler to ensure all features contribute equally to distance-based algorithms like SVR and PCA.

Train-Test Split

Data split into 80% training and 20% testing for fair model evaluation.

# 5. Machine Learning Models Used
## 5.1 Linear Regression

A baseline statistical model that assumes a linear relationship between predictors and life expectancy.

Pros: Simple, interpretable, and fast.

Cons: Limited ability to capture non-linear relationships.

Use case: Establishes a baseline performance metric to compare advanced models against.

## 5.2 Support Vector Regressor (SVR)

SVR attempts to find an optimal hyperplane that fits the data within a specified margin of error.

Pros: Effective in high-dimensional spaces and robust to outliers.

Cons: Sensitive to feature scaling and can be slow for large datasets.

Use case: Captures more complex, non-linear trends that linear regression may miss.

## 5.3 PCA + SVR Pipeline

Principal Component Analysis (PCA) reduces the dimensionality of the data while retaining 95% of the variance, followed by SVR on the compressed features.

Pros: Reduces noise, overfitting, and computational cost.

Cons: Slightly reduces interpretability.

Use case: A hybrid model balancing performance and efficiency, particularly useful when dealing with correlated health and economic features.

## 5.4 Random Forest Regressor

An ensemble learning method that builds multiple decision trees and averages their outputs.

Pros: Handles non-linear data, robust to outliers, and provides feature importance.

Cons: Can be computationally heavier than simpler models.

Use case: The most accurate and stable model for this dataset. It also provides valuable interpretability through feature importance ranking.

# 6. Model Evaluation Metrics
Metric	Description	Ideal Value
R² Score	Measures how much of the variance in life expectancy the model can explain	Higher values indicate better performance
Mean Absolute Error (MAE)	Represents the average prediction error in years	Lower values indicate higher precision
Final Observations

Random Forest achieved the highest R² and lowest MAE, demonstrating its superior predictive capability.

Linear Regression served as a strong baseline but underperformed in capturing non-linear dependencies.

PCA + SVR provided balanced results with improved efficiency.

# 7. Visualizations

Model Performance Comparison: Dual bar charts showing R² and MAE for each model to visualize trade-offs between accuracy and error.

Actual vs Predicted (Random Forest): Scatter plot showing how close predictions were to actual life expectancy values.

Feature Importance: Ranked bar chart displaying the most influential predictors of life expectancy, such as “Schooling”, “Adult Mortality”, and “Income Composition of Resources”.

# 8. Streamlit Web Application

A lightweight, interactive web application was built using Streamlit to make the model accessible and user-friendly.

Features

Accepts key inputs such as GDP, schooling, BMI, and HIV/AIDS rates.

Converts the data into model-readable format using the same scaler as in training.

Displays predicted Life Expectancy instantly upon user submission.

Objective: To demonstrate real-world usability of the trained model and allow users to simulate life expectancy under various conditions.

# 9. Key Insights

Education level and income composition have strong positive effects on life expectancy.

High adult mortality, HIV/AIDS prevalence, and under-five deaths strongly reduce life expectancy.

Developing nations generally exhibit greater variability due to wider economic and healthcare disparities.

# 10. Tools and Technologies
Category	Tools Used
Programming Language	Python
Libraries	pandas, numpy, matplotlib, seaborn, scikit-learn
Model Deployment	Streamlit
Data Handling	StandardScaler, PCA
Evaluation Metrics	R², MAE
# 11. Challenges Faced and How They Were Overcome
Data Cleaning and Preprocessing

Issue: The raw dataset contained missing values in several columns such as GDP, Alcohol, and Hepatitis B, along with inconsistent column names.
Fix: Missing values were replaced using each column’s median. Column names were standardized, and categorical columns were encoded using pd.get_dummies().

Model Selection and Evaluation

Issue: It was unclear which algorithm would best predict life expectancy.
Fix: Implemented and compared Linear Regression, SVR, Random Forest, and PCA + SVR pipeline using R² and MAE. Random Forest performed best (R² ≈ 0.95, MAE ≈ 1.13 years).

Feature Scaling and Compatibility

Issue: Some models required scaled data, others didn’t.
Fix: Used StandardScaler for all models to ensure uniformity.

Implementing PCA + SVR Pipeline

Issue: Shape mismatch errors and parameter tuning difficulties.
Fix: Used Scikit-learn’s Pipeline to combine PCA and SVR seamlessly (n_components=0.95).

Model Deployment and Input Matching

Issue: Feature name and column order mismatches between training and Streamlit input.
Fix: Used .reindex(columns=original_columns) to align user input with the trained model.

Visualization Challenges

Issue: Initial plots were cluttered and unreadable.
Fix: Improved visualization clarity with Seaborn and Matplotlib (adjusted sizes, palettes, and annotations).

# 12. Learning Outcomes
Data Preprocessing Skills

Cleaned real-world datasets, handled missing values, and encoded categorical variables.

Understood the importance of normalization and feature selection.

Model Building and Evaluation

Trained multiple regression models and evaluated them using R² and MAE.

Learned trade-offs between linear and non-linear models.

Feature Importance Analysis

Interpreted Random Forest feature importance to identify top predictors (Income composition, HIV/AIDS, Adult Mortality).

Model Deployment with Streamlit

Converted an ML model into an interactive web app.

Connected user input fields to backend prediction logic.

End-to-End Project Experience

Completed full ML workflow: data cleaning, training, testing, visualization, model saving, and deployment.

Improved debugging and version compatibility for deployment.

# 13. Conclusion

This project demonstrates how data-driven models can provide actionable insights into public health outcomes.
Among all tested models, Random Forest Regression achieved the best balance of accuracy and interpretability.
The Streamlit app extends this analysis into an interactive tool for real-world exploration and prediction.

Future Work:

Include more granular features such as healthcare spending and pollution indices.

Deploy the model as an API integrated with live global datasets.


# Steps to run Code:

How to Run This Project Locally

You can run the interactive Streamlit app on your own computer by following these steps.

## 1. Clone the Repository
   
First, get the code from GitHub. Open your terminal and run:
bash
git clone [https://github.com/virxnyu/life-expectancy-predictor.git](https://github.com/virxnyu/life-expectancy-predictor.git)
cd life-expectancy-predictor

## 2. Create a Virtual Environment
   
This is a best practice to keep all the project libraries in one place.

### Windows
python -m venv venv
.\venv\Scripts\activate

### macOS / Linux
python3 -m venv venv
source venv/bin/activate

## 3. Install all required libraries from requirements.txt
   
Run pip install -r requirements.txt to install all libraries

## 4. Run the Streamlit App

streamlit run app.py


