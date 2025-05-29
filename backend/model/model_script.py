import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import r2_score
import numpy as np
import pickle
import os

# === Paths ===
file_path = 'C:/Users/harin/Downloads/gdp-X-main/gdp-X-main/backend/model/india_gdp/gdp.csv'
model_dir = 'C:/Users/harin/Downloads/gdp-X-main/gdp-X-main/backend/model'

# === Load & Preprocess ===
try:
    data = pd.read_csv(file_path, skiprows=4)
    print("✅ Dataset loaded successfully!")

    gdp_data = data[
        (data['Country Name'] == 'India') & 
        (data['Indicator Name'] == 'GDP (current US$)')
    ].drop(columns=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'], errors='ignore')

    gdp_data = gdp_data.melt(var_name='year', value_name='gdp')
    gdp_data['year'] = pd.to_numeric(gdp_data['year'], errors='coerce')
    gdp_data['gdp'] = pd.to_numeric(gdp_data['gdp'], errors='coerce')
    gdp_data = gdp_data.dropna()
    print("✅ Data cleaned and reshaped successfully!")

    # Log-transform GDP to handle large values
    gdp_data['log_gdp'] = np.log(gdp_data['gdp'])

    # === Features & Labels ===
    X = gdp_data[['year']].values
    y = gdp_data['log_gdp'].values

    # === Train-Test Split ===
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # === Model ===
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_score = r2_score(y_train, y_train_pred)
    test_score = r2_score(y_test, y_test_pred)

    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='r2')

    print(f"✅ Training R² Score: {train_score:.4f}")
    print(f"✅ Testing R² Score: {test_score:.4f}")
    print(f"✅ Mean TimeSeries CV R² Score: {cv_scores.mean():.4f}")

    # === Save Model ===
    model_path = os.path.join(model_dir, 'gdp_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    print("🎉 Model saved successfully!")

except Exception as e:
    print(f"❌ Error: {e}")
