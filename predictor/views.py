from django.shortcuts import render
import pickle
import numpy as np
import os
import json
# NEW: Import the BASE_DIR setting from your project's settings
from crop_project.settings import BASE_DIR

# --- 1. Load Models using RELATIVE paths ---
# This will now work on your computer AND on the Render server
model_path = os.path.join(BASE_DIR, 'crop_yield_model.pkl')
scaler_path = os.path.join(BASE_DIR, 'scaler.pkl')
encoders_path = os.path.join(BASE_DIR, 'encoders.pkl')

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

# The rest of your 'predict' function does not need to change.
# It is included here for completeness.
def predict(request):
    context = {
        'crop_list': CROP_LIST,
        'season_list': SEASON_LIST,
        'state_list': STATE_LIST,
    }

    if request.method == 'POST':
        try:
            crop = request.POST.get('crop')
            season = request.POST.get('season')
            state = request.POST.get('state')
            rainfall = float(request.POST.get('rainfall'))
            fertilizer = float(request.POST.get('fertilizer'))
            pesticide = float(request.POST.get('pesticide'))

            crop_encoded = encoders['Crop'].transform([crop])[0]
            season_encoded = encoders['Season'].transform([season])[0]
            state_encoded = encoders['State'].transform([state])[0]

            features_list = [
                crop_encoded, season_encoded, state_encoded,
                rainfall, fertilizer, pesticide
            ]
            features_scaled = scaler.transform([features_list])
            prediction_value = model.predict(features_scaled)[0]
            context['prediction_text'] = f"Predicted Crop Yield: {prediction_value:.2f} tons/hectare"

            feature_names = ['Crop', 'Season', 'State', 'Annual Rainfall', 'Fertilizer', 'Pesticide']
            importances = model.feature_importances_
            feature_importance_dict = dict(zip(feature_names, importances))
            sorted_features = sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True)
            
            chart_labels = [item[0] for item in sorted_features]
            chart_data = [float(item[1]) for item in sorted_features]

            context['chart_labels'] = json.dumps(chart_labels)
            context['chart_data'] = json.dumps(chart_data)

        except Exception as e:
            context['prediction_text'] = f"Error processing request: {e}"

    return render(request, 'predictor/index.html', context)
