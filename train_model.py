import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv('/teamspace/studios/this_studio/car_price_prediction/car data.csv')

# Preprocessing
df.drop('Car_Name', axis=1, inplace=True)
df['Car_Age'] = 2025 - df['Year']
df.drop('Year', axis=1, inplace=True)
df = pd.get_dummies(df, drop_first=True)

# Features & Labels
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open('model.pkl', 'wb'))
print("✅ Model saved as model.pkl")
