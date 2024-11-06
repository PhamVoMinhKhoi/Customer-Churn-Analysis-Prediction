# Customer Churn Analysis Prediction

This project aims to analyze customer churn data to identify factors that contribute to customer churn and to build a predictive model for determining the likelihood of churn among customers. By understanding and predicting churn, businesses can take preventive actions to retain customers.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset Information](#dataset-information)
3. [Analysis and Key Metrics](#analysis-and-key-metrics)
4. [Modeling Approach](#modeling-approach)
5. [Results and Conclusion](#results-and-conclusion)
6. [Usage](#usage)
7. [License](#license)

---

## Project Overview

Customer churn refers to the loss of clients or customers. It’s a critical metric for companies to monitor, as high churn rates indicate dissatisfaction or unmet customer needs. This project:
- Analyzes key factors affecting churn rates.
- Builds a predictive model to classify if a customer is likely to churn.
- Helps businesses prioritize retention efforts.

## Dataset Information

The dataset used includes a variety of customer attributes:
- **Demographics**: Customer tenure, contract type, payment method, etc.
- **Usage**: Monthly charges, total charges, etc.
- **Services**: Internet service, phone service, and additional services such as streaming or online security.

Each row represents a customer, and the target variable is **Churn** (indicating whether the customer has left the company).

## Analysis and Key Metrics

Key metrics analyzed include:

1. **Customer Tenure**: Represents the length of time a customer has been with the company.
2. **Monthly Charges and Total Charges**: Indicate the financial relationship between the customer and the company.
3. **Contract Type**: Different contract types have varying impacts on churn rates (e.g., month-to-month contracts are associated with higher churn).
4. **Service Usage**: Use of services like internet, phone, and streaming can influence customer satisfaction and loyalty.
5. **Payment Method**: Examined to see if automatic payments vs. manual payments impact churn.

### Observations:
- Higher churn rates are often observed in customers with month-to-month contracts, higher monthly charges, and no additional services.
- Lower churn rates are seen in customers with longer tenure and bundled services.

## Modeling Approach

The modeling process includes:

1. **Data Preprocessing**: Cleaning, encoding categorical variables, and feature scaling.
2. **Feature Engineering**: Selecting relevant features based on correlation analysis.
3. **Model Selection and Evaluation**: Models tested include Logistic Regression, Decision Tree, Random Forest, and XGBoost. The evaluation metrics include:
   - **Accuracy**: The percentage of correctly predicted cases.
   - **Precision and Recall**: Indicators of the model's effectiveness in identifying true churn cases.
   - **F1 Score**: A balance between precision and recall.
   - **ROC-AUC**: Measures the model's ability to distinguish between churn and non-churn.

## Results and Conclusion

- The **Random Forest** and **XGBoost** models performed the best, achieving high accuracy and a strong ROC-AUC score.
- **Key Features**: Monthly charges, contract type, and tenure are significant indicators of churn likelihood.
  
The final model provides valuable insights into factors driving customer churn, helping businesses create targeted retention strategies.

## Usage

To replicate or extend the project:

1. Clone this repository.
2. Ensure you have the necessary libraries installed (e.g., `pandas`, `scikit-learn`, `xgboost`).
3. Load the dataset and execute the notebook cells in order.

```python
# Example code snippet to train and test the model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
