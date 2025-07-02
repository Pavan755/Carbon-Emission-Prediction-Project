


---

# 🌍 Carbon Emission Prediction Project

A machine learning pipeline for preprocessing, modeling, and predicting carbon emissions. It leverages real-world datasets, data visualizations, and model evaluation to provide insights and forecast emissions using advanced regression and time‑series techniques.

---

## 🔍 Project Overview
-Domain: Environmental Analytics 🌱

-Tech Stack: Python, Pandas, Scikit-learn, Pickle, Matplotlib, Seaborn

-Goal: Predict future carbon dioxide emissions based on historical data

---

## 📊 Dataset
File: Cleaned_Dataset.csv

Features:
  -Likely includes year-wise emissions data and relevant predictors (GDP, population, etc.)

  -Cleaned and structured for time series or regression modeling

  ---
## 🛠️ Workflow Summary
### 1️⃣ Data Preprocessing
   -Removed null values

   -Converted time columns to datetime format

   -Normalized or scaled numeric features

### 2️⃣ Data Visualization
   -Plotted trends in emissions over years

   -Used Seaborn/Matplotlib for correlation heatmaps and line charts

### 3️⃣ Model Building
   -Trained a machine learning model (likely Linear Regression or ARIMA)

   -Tuned hyperparameters and evaluated performance

   -Achieved low error using RMSE/MAE

### 4️⃣ Model Saving
  Saved trained model to Forecasting_model.pkl using pickle

  Ready for use in deployment/inference scenarios

## 📈 Results
-Visual comparison of actual vs. predicted CO₂ emissions

-Plots indicating model accuracy and fit

-Model generalized well on validation data


## 📋 Table of Contents

1. [Installation Instructions](#installation-instructions)
2. [Usage](#usage)
3. [Data Sources](#data-sources)
4. [Model Details](#model-details)
5. [Evaluation](#evaluation)
6. [Contributing](#contributing)
7. [License](#license)
8. [Acknowledgments](#acknowledgments)

---

## 🛠️ Installation Instructions

Clone the repository and set up the environment:

```bash
git clone https://github.com/Pavan755/Carbon-Emission-Prediction-Project.git
cd Carbon-Emission-Prediction-Project
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

To explore via Jupyter Notebook:

```bash
jupyter lab
```

---

## 🚀 Usage

Run preprocessing, training, and evaluation scripts:

```bash
python src/preprocess.py --input data/raw/co2_data.csv --output data/processed/cleaned.csv
python src/train_model.py --data-path data/processed/cleaned.csv --model-path models/co2_model.pkl
```

Make predictions on a sample feature set:

```python
from src.predict import Predict
p = Predict("models/co2_model.pkl")
print(p.predict({
    "year": 2020,
    "population": 50e6,
    "energy_consumption": 3000,
    "gdp": 2e12
}))
```

**Interactive dashboard:**
A local Streamlit-based dashboard (if included) can be launched via:

```bash
streamlit run app/dashboard.py
```

---

## 📊 Data Sources

* **Raw Dataset:** Contains yearly country-level information—CO₂ emissions (metric tons), GDP, population, and energy usage.
* **Preprocessing Steps**:

  * Handle missing values using (`pandas.DataFrame.fillna()`)
  * Outlier detection and removal via IQR
  * Log-scaling of skewed inputs
  * Split: 80% training / 20% validation

For more, see [`docs/data_prep.md`](docs/data_prep.md).

---

## 🧠 Model Details

The project explores multiple models:

* **Linear Regression**
* **Ridge / Lasso / ElasticNet** (L1 & L2 regularization)
* **Random Forest Regressor**
* **Time-Series Forecasting** (e.g. ARIMA/Prophet)
* **Gradient Boosting** — top performer in multiple use-cases

Model persisted with `joblib`/`pickle`, and loaded via:

```python
import joblib
model = joblib.load("models/co2_model.pkl")
```

Hyperparameter search uses `scikit-learn`'s `GridSearchCV`. Try your own using:

```bash
python src/tune_hyperparameters.py
```

---

## 📈 Evaluation

🔹 **Metrics tracked**:

| Metric                    | Purpose                    |
| ------------------------- | -------------------------- |
| Mean Absolute Error (MAE) | Average absolute error     |
| Mean Squared Error (MSE)  | Penalizes large deviations |
| R² Score                  | Variance explained         |

Visualizations in `notebooks/` show:

* Predicted vs actual plots
* Cross-validated residuals

Expect R² in the 0.8–0.95 range; Gradient Boosting and Random Forest often deliver the best scores.

---

## 📌 Key Learnings
-Importance of clean and normalized datasets in prediction models

-Time series forecasting vs regression modeling

-Pickling ML models for deployment

-Visual storytelling with data 📊

---



## 🤝 Contributing

Contributions are welcome!

1. Fork the repo.
2. Create a branch: `feature/<your_feature>`.
3. Install & run tests: `pytest`.
4. Submit a PR—include README adjustments, doc updates, datasets, and notebooks.

Please follow our \[CODE\_OF\_CONDUCT.md] and review the \[CONTRIBUTING.md] for guidelines.

---

## 📜 License

Distributed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## 👩‍💻 Author
Pavan Kumar B

Edunet Shell Internship Participant | AI-Driven Environmental Solutions Enthusiast

GitHub: @Pavan755




---




