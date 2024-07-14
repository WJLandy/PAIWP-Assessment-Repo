import pandas as pd

# Load your dataset
data = pd.read_csv('Customer_Churn.csv')

# Check the distribution of the target variable in the original data
print("Original Churn distribution:")
print(data['Churn'].value_counts())

# Preprocess the data
# Handling missing values
data = data.dropna()

# Check for any whitespace values and replace them with NaN
data.replace(' ', pd.NA, inplace=True)

# Drop rows with any remaining missing values
data.dropna(inplace=True)

# Verify that 'Churn' is still 'Yes'/'No' before transformation
print("Churn column before transformation:")
print(data['Churn'].unique())

# Convert 'Churn' column to numeric values if not already in 0/1 format
if data['Churn'].dtype == object:
    data['Churn'] = data['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# Verify the target variable transformation
print("Transformed Churn distribution:")
print(data['Churn'].value_counts())

# Encoding categorical variables
categorical_features = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
                        'PaperlessBilling', 'PaymentMethod']

# Apply one-hot encoding to categorical features
data = pd.get_dummies(data, columns=categorical_features, drop_first=True)

# Verify data after encoding
print("Data after encoding:")
print(data.head())

# Save the cleaned data to a new CSV file
data.to_csv('cleaned_data.csv', index=False)