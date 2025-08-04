# 🌾 Crop Yield Prediction 

## 📌 Project Description

This project predicts crop yield (in tonnes/hectare) based on historical data including weather (rainfall, temperature), soil properties (pH, nitrogen, etc.), and crop type. It uses machine learning models and includes a web-based frontend for user interaction and data visualization.

---

## 🎯 Objectives

- Predict yield based on soil and weather parameters.
- Enable user-friendly input via frontend.
- Display predictions and data insights using visualizations.
- Aid farmers and agriculture planners in better decision-making.

---

## 🧰 Tech Stack

| Area          | Technology                      |
|---------------|----------------------------------|
| Programming   | Python 3                         |
| Machine Learning | scikit-learn, pandas, NumPy    |
| Visualization | matplotlib, seaborn, Plotly      |
| Backend       | Flask                            |
| Frontend      | HTML, CSS, JavaScript (Bootstrap)|
| Model         | Linear Regression / Random Forest|
| Deployment    | Localhost / Flask server         |

---

## 📂 Dataset Details

1. `crop_data.csv`
   - Columns:
     - Crop
     - Rainfall (mm)
     - Temperature (°C)
     - pH
     - Nitrogen (N)
     - Phosphorus (P)
     - Potassium (K)
     - Yield (tonnes/hectare)

> 📌 Note: Dataset can be collected from [Kaggle](https://www.kaggle.com), ICAR (India), or FAO sources.

---

## 🚀 Features

- ✅ Crop yield prediction using trained ML model
- ✅ Frontend form for user input
- ✅ Visualization of dataset and predictions
- ✅ Downloadable prediction report (optional)
- ✅ Responsive UI using Bootstrap

---

## 🖼️ Sample Visualizations

- **Correlation Heatmap**
- **Crop-wise Yield Bar Plot**
- **Scatter Plot: Rainfall vs Yield**

---

## 🧪 How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/crop-yield-predictor.git
cd crop-yield-predictor
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Flask App

```bash
python app.py
```

### 4. Open in Browser

Visit: `http://127.0.0.1:5000`

---

## 🗂️ Project Structure

```
├── static/
│   └── style.css
├── templates/
│   ├── index.html
│   ├── result.html
│   └── visualize.html
├── crop_data.csv
├── app.py
├── model.pkl
├── requirements.txt
└── README.md
```

---

## 🧠 Machine Learning Model

- **Model Used**: Random Forest Regressor (can switch to Linear Regression)
- **Accuracy**: ~90% (varies with dataset)
- **Target Variable**: Yield (tonnes/hectare)

---

## 📊 Frontend Preview

- **Home Page**: Enter soil, weather, crop details
- **Prediction Page**: Displays predicted yield
- **Visualization Page**: Shows charts (bar, heatmap, scatter)

---

## 🛠️ Sample Input (From UI)

```
Crop: Rice
Rainfall: 120 mm
Temperature: 28°C
pH: 6.5
N: 80
P: 45
K: 40
```

### ✅ Sample Output:
```
Predicted Yield: 3.8 tonnes/hectare
```

---

## 📈 Sample Charts

1. Bar chart of yield by crop
2. Heatmap showing correlation of features
3. Scatter plot for rainfall vs yield

---

## 👤 Author

- **Name**: Charan Basava
- **GitHub**: [CharanBasava](https://github.com/CharanBasava)

---

## 📄 License

This project is open-source and available under the MIT License.

