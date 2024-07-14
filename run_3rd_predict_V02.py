import pandas as pd
import joblib
import shap
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the saved model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Load and preprocess new data
new_data = pd.read_csv('new_data.csv')

# Check for any whitespace values and replace them with NaN
new_data.replace(' ', pd.NA, inplace=True)

# Drop rows with any remaining missing values
new_data.dropna(inplace=True)

# Encode categorical variables
categorical_features = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
                        'PaperlessBilling', 'PaymentMethod']

# Apply one-hot encoding to categorical features
new_data = pd.get_dummies(new_data, columns=categorical_features, drop_first=True)

# Ensure new data has the same columns as the training data
# You can use the training data to get the correct columns
training_columns = pd.read_csv('cleaned_data.csv').drop(['customerID', 'Churn'], axis=1).columns
for col in training_columns:
    if col not in new_data.columns:
        new_data[col] = 0  # Add missing columns with default value 0

new_data = new_data[training_columns]  # Reorder columns to match training data

# Feature scaling
new_data_scaled = scaler.transform(new_data)

# Make predictions
predictions = model.predict(new_data_scaled)
predictions_proba = model.predict_proba(new_data_scaled)[:, 1]  # Probability of class 1

# Explain predictions with SHAP
explainer = shap.Explainer(model, new_data_scaled)
shap_values = explainer(new_data_scaled)

# Convert SHAP values to a DataFrame for easier handling
shap_df = pd.DataFrame(shap_values.values, columns=new_data.columns)
shap_df['Churn_Prediction'] = predictions
shap_df['Churn_Probability'] = predictions_proba

# Combine new_data with SHAP values for output
output_df = new_data.copy()
output_df['Churn_Prediction'] = predictions
output_df['Churn_Probability'] = predictions_proba

# Add SHAP values for each feature
for column in shap_df.columns[:-2]:  # Exclude 'Churn_Prediction' and 'Churn_Probability'
    output_df[f'SHAP_{column}'] = shap_df[column]

# Save the predictions and SHAP values to a new CSV file
output_df.to_csv('predictions_with_shap.csv', index=False)

print(output_df[['Churn_Prediction', 'Churn_Probability']])