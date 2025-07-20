# 🚗 Car Price Prediction

A machine learning web app built using **Flask** that predicts the **selling price of a car** based on features like present price, kms driven, fuel type, transmission, ownership, etc.

## 📁 Project Structure

```
car-price-prediction/
│
├── model.pkl               # Trained Random Forest model
├── app.py                  # Flask backend app
├── templates/
│   └── index.html          # Main HTML form page
├── static/
│   └── style.css           # Styling for the page
├── car data.csv            # Dataset used for training
└── README.md               # Project overview
```

## 🚀 Tech Stack

- Python
- Pandas, Scikit-learn
- Flask
- HTML/CSS (basic UI)

## 📊 Dataset

Dataset contains the following columns:

- `Car_Name`
- `Year`
- `Selling_Price`
- `Present_Price`
- `Driven_kms`
- `Fuel_Type`
- `Selling_type`
- `Transmission`
- `Owner`

## 🧠 Model Training

```python
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('car data.csv')
df.drop('Car_Name', axis=1, inplace=True)
df['Car_Age'] = 2025 - df['Year']
df.drop('Year', axis=1, inplace=True)
df = pd.get_dummies(df, drop_first=True)

X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(X_train, y_train)

pickle.dump(model, open('model.pkl', 'wb'))
```

## 🌐 Run the Flask App

```bash
python app.py
```

By default, it will run on `http://127.0.0.1:5000`



## ✅ Features

- Predicts used car prices instantly
- Minimal and clean UI
- Supports dropdowns and form validations

## 📌 Future Scope

- Add login/signup authentication
- Visualize price trends with graphs
- Deploy to cloud (Render, Vercel, etc.)

---

Made with ❤️ by [Ajay Chaudhary](https://github.com/ajaychaudhary2)
