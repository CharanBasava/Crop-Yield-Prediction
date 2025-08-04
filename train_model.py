import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from xgboost import XGBRegressor
import pickle
import os

print("--- Model Training Script Started ---")

# --- 1. Define Paths and Load Data ---
base_dir = r'C:\Users\Basava Charan\Desktop\crop_yield_prediction'
dataset_path = os.path.join(base_dir, 'datasets', 'crop_yield_data.csv')

try:
    df = pd.read_csv(dataset_path)
    print(f"Successfully loaded dataset from: {dataset_path}")
except FileNotFoundError:
    print(f"ERROR: Dataset not found at {dataset_path}")
    print("Please make sure the file 'crop_yield_data.csv' is in the 'datasets' folder.")
    exit()

# --- 2. Data Cleaning and Preprocessing ---
# Drop rows with any missing values
df.dropna(inplace=True)
print("Dropped rows with missing values.")

# Encode all categorical features: Crop, Season, and State
categorical_cols = ['Crop', 'Season', 'State']
encoders = {col: LabelEncoder() for col in categorical_cols}

for col in categorical_cols:
    df[col] = encoders[col].fit_transform(df[col])

print("Encoded categorical columns: Crop, Season, State.")

# --- 3. Define Features and Target ---
# **This is the updated list of features to match your new dataset**
features = ['Crop', 'Season', 'State', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']
target = 'Yield'

X = df[features]
y = df[target]
print(f"Using features: {features}")

# Scale numerical features to a range of 0-1
# We will fit the scaler on the entire feature set X
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# --- 4. Train the XGBoost Model ---
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print("Training the XGBoost model...")
model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)
model.fit(X_train, y_train)
print("Model training complete.")

# --- 5. Evaluate Model Performance ---
score = model.score(X_test, y_test)
print(f"Model R² Score (a measure of accuracy): {score:.2f}")

# --- 6. Save the Trained Components ---
pickle.dump(model, open(os.path.join(base_dir, 'crop_yield_model.pkl'), 'wb'))
pickle.dump(scaler, open(os.path.join(base_dir, 'scaler.pkl'), 'wb'))
# Save the dictionary of encoders
pickle.dump(encoders, open(os.path.join(base_dir, 'encoders.pkl'), 'wb'))

print("\nSuccessfully saved the following files in your project directory:")
print(f"- crop_yield_model.pkl (The trained model)")
print(f"- scaler.pkl (To process user input)")
print(f"- encoders.pkl (To handle all categorical inputs)")
print("--- Script Finished ---")
