import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Load dataset
data = pd.read_csv('Customer_Churn.csv')

# Preprocess data
# Example: Handling missing values
data = data.dropna()

# Encoding categorical variables
categorical_features = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
                        'PaperlessBilling', 'PaymentMethod']
data = pd.get_dummies(data, columns=categorical_features, drop_first=True)

# Splitting the data
X = data.drop(['customerID', 'Churn'], axis=1)  # Features
y = data['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)  # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
new_data = pd.read_csv('/mnt/data/new_data.csv')
new_data = pd.get_dummies(new_data, columns=categorical_features, drop_first=True)
new_data = scaler.transform(new_data)  # Ensure new data is scaled similarly
predictions = model.predict(new_data)

# Output predictions for new data
print(predictions)
