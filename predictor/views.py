from django.shortcuts import render
import pickle
import numpy as np
import os
import json # Import the json library

# --- 1. Load the Saved Models and Encoders ---
base_dir = r'C:\Users\Basava Charan\Desktop\crop_yield_prediction'
model_path = os.path.join(base_dir, 'crop_yield_model.pkl')
scaler_path = os.path.join(base_dir, 'scaler.pkl')
encoders_path = os.path.join(base_dir, 'encoders.pkl')

try:
    model = pickle.load(open(model_path, 'rb'))
    scaler = pickle.load(open(scaler_path, 'rb'))
    encoders = pickle.load(open(encoders_path, 'rb'))
    print("Model and supporting files loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading files: {e}.")
    model, scaler, encoders = None, None, None

CROP_LIST = sorted(list(encoders['Crop'].classes_)) if encoders else []
SEASON_LIST = sorted(list(encoders['Season'].classes_)) if encoders else []
STATE_LIST = sorted(list(encoders['State'].classes_)) if encoders else []


def predict(request):
    context = {
        'crop_list': CROP_LIST,
        'season_list': SEASON_LIST,
        'state_list': STATE_LIST,
    }

    if request.method == 'POST':
        try:
            # --- Get User Input (Same as before) ---
            crop = request.POST.get('crop')
            season = request.POST.get('season')
            state = request.POST.get('state')
            rainfall = float(request.POST.get('rainfall'))
            fertilizer = float(request.POST.get('fertilizer'))
            pesticide = float(request.POST.get('pesticide'))

            # --- Transform and Scale (Same as before) ---
            crop_encoded = encoders['Crop'].transform([crop])[0]
            season_encoded = encoders['Season'].transform([season])[0]
            state_encoded = encoders['State'].transform([state])[0]

            features_list = [
                crop_encoded, season_encoded, state_encoded,
                rainfall, fertilizer, pesticide
            ]
            features_scaled = scaler.transform([features_list])

            # --- Make Prediction (Same as before) ---
            prediction_value = model.predict(features_scaled)[0]
            context['prediction_text'] = f"Predicted Crop Yield: {prediction_value:.2f} tons/hectare"

            # --- NEW: Get Feature Importances for the Chart ---
            # Get the feature names in the correct order
            feature_names = ['Crop', 'Season', 'State', 'Annual Rainfall', 'Fertilizer', 'Pesticide']
            # Get importance scores from the trained model
            importances = model.feature_importances_

            # Create a dictionary of feature names and their importance scores
            feature_importance_dict = dict(zip(feature_names, importances))
            
            # Sort the features by importance for a better looking chart
            sorted_features = sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True)
            
            # Prepare data for Chart.js
            chart_labels = [item[0] for item in sorted_features]
            chart_data = [float(item[1]) for item in sorted_features] # Ensure data is float

            # Add chart data to the context using json.dumps
            context['chart_labels'] = json.dumps(chart_labels)
            context['chart_data'] = json.dumps(chart_data)


        except Exception as e:
            context['prediction_text'] = f"Error processing request: {e}"

    return render(request, 'predictor/index.html', context)

