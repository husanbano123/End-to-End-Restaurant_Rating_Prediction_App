import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import pickle

# Load the dataset
df = pd.read_csv('Restaurant_Rating_Data.csv')  # Replace with your actual CSV file

# Selecting features and target variable
X = df[['Delivery', 'Booking', 'Votes', 'Location', 'Rest_Type', 'Cuisines', 'Cost_of_Two_People', 'Type']]
y = df['Rating']

# Label encoding categorical features
label_encoders = {}
for column in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    label_encoders[column] = le  # Storing the label encoder for possible inverse_transform later

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)


# Save the model and label encoders
with open('restaurant_rating_model.pkl', 'wb') as file:
    pickle.dump({'model': model, 'label_encoders': label_encoders}, file)