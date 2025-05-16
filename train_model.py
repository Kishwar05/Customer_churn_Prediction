import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load data
df = pd.read_csv('data/Telco-Customer-Churn.csv')

# Clean and preprocess
df = df[df['TotalCharges'] != ' ']
df['TotalCharges'] = df['TotalCharges'].astype(float)
df.dropna(inplace=True)

# Encode categorical variables and save encoders
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    if col != 'customerID':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

X = df.drop(['customerID', 'Churn'], axis=1)
y = df['Churn']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(class_weight='balanced')
model.fit(X_train, y_train)

# Save model, encoders, and feature list
with open('model/churn_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('model/label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
with open('model/feature_columns.pkl', 'wb') as f:
    pickle.dump(list(X.columns), f)
