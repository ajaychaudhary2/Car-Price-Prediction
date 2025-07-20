import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# ✅ Step 1: Load your CSV (adjust path if needed)
df = pd.read_csv('car data.csv')  # path should match your local file

# ✅ Step 2: Preprocessing
df.drop('Car_Name', axis=1, inplace=True)
df['Car_Age'] = 2025 - df['Year']
df.drop('Year', axis=1, inplace=True)
df = pd.get_dummies(df, drop_first=True)

# ✅ Step 3: Split features & label
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

# ✅ Step 4: Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor()
model.fit(X_train, y_train)

# ✅ Step 5: Overwrite the old model.pkl
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ model.pkl retrained and saved with correct sklearn version!")
