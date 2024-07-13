import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Load your dataset
data = pd.read_csv('Customer_Churn.csv')

# Check the distribution of the target variable in the original data
print("Original Churn distribution:")
print(data['Churn'].value_counts())

# Preprocess the data
# Example: Handling missing values
data = data.dropna()

# Check for any whitespace values and replace them with NaN
data.replace(' ', pd.NA, inplace=True)

# Drop rows with any remaining missing values
data.dropna(inplace=True)

# Encoding categorical variables
categorical_features = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
                        'PaperlessBilling', 'PaymentMethod']

# Verify that 'Churn' is still 'Yes'/'No' before transformation
print("Churn column before transformation:")
print(data['Churn'].unique())

# Convert 'Churn' column to numeric values
data['Churn'] = data['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# Verify the target variable transformation
print("Transformed Churn distribution:")
print(data['Churn'].value_counts())

# Apply one-hot encoding to categorical features
data = pd.get_dummies(data, columns=categorical_features, drop_first=True)

# Verify data after encoding
print("Data after encoding:")
print(data.head())

# Splitting the data with stratified sampling
X = data.drop(['customerID', 'Churn'], axis=1)  # Features
y = data['Churn']  # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Check the distribution in the training and test sets
print("Training set Churn distribution:")
print(y_train.value_counts())
print("Test set Churn distribution:")
print(y_test.value_counts())

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model selection and training
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print('ROC AUC Score:', roc_auc_score(y_test, y_pred))

# Make predictions on new data
new_data = pd.read_csv('new_data.csv')
new_data.replace(' ', pd.NA, inplace=True)
new_data.dropna(inplace=True)
new_data = pd.get_dummies(new_data, columns=categorical_features, drop_first=True)
new_data = new_data.apply(pd.to_numeric, errors='coerce')
new_data = scaler.transform(new_data)  # Ensure new data is scaled similarly
predictions = model.predict(new_data)

# Output predictions for new data
print(predictions)