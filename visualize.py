import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

# Use non-interactive backend
matplotlib.use('Agg')

def generate_gdp_plot(year):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'backend', 'model', 'india_gdp', 'gdp.csv')
    model_path = os.path.join(base_dir, 'backend', 'model', 'gdp_model.pkl')

    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Load GDP data
    data = pd.read_csv(data_path, skiprows=4)
    gdp_data = data[data['Indicator Name'] == 'GDP (current US$)']
    gdp_data = gdp_data.drop(columns=[
        'Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'
    ], errors='ignore')
    gdp_data = gdp_data.melt(var_name='year', value_name='gdp').dropna()
    gdp_data['year'] = pd.to_numeric(gdp_data['year'], errors='coerce')
    gdp_data['gdp'] = pd.to_numeric(gdp_data['gdp'], errors='coerce')
    gdp_data = gdp_data.dropna()

    # Predict GDP (log to real)
    predicted_log_gdp = model.predict([[year]])[0]
    predicted_gdp = np.exp(predicted_log_gdp)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(gdp_data['year'], gdp_data['gdp'], color='green', label='Actual GDP')
    plt.scatter([year], [predicted_gdp], color='red', s=100, label=f'Predicted GDP for {year}')
    plt.xlabel('Year')
    plt.ylabel('GDP (Current US$)')
    plt.title(f'GDP Prediction for {year}')
    plt.legend()
    plt.grid(True)
    plt.xlim(gdp_data['year'].min(), max(gdp_data['year'].max(), year + 1))

    # Save Plot
    image_dir = os.path.join(base_dir, 'static', 'images')
    os.makedirs(image_dir, exist_ok=True)
    plot_path = os.path.join(image_dir, 'plot.png')
    plt.savefig(plot_path)
    plt.close()

    print(f"[✔] Predicted GDP for {year}: {predicted_gdp:.2f} USD")
    print(f"[✔] Plot saved at: {plot_path}")
    return predicted_gdp, plot_path


# ------------------------------
# 🧠 User Input
# ------------------------------
if __name__ == "__main__":
    try:
        year_input = int(input("📅 Enter the year for prediction (e.g., 2026): "))
        predicted_gdp, _ = generate_gdp_plot(year_input)
    except Exception as e:
        print(f"[❌] Error: {e}")
