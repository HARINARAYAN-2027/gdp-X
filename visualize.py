import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os

# Save graph without showing it
matplotlib.use('Agg') 

def generate_gdp_plot(year, predicted_gdp):
    # Correct absolute path for the CSV file
    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, 'model', 'india_gdp', 'gdp.csv')

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at: {file_path}")

    # Load and process GDP data
    data = pd.read_csv(file_path, skiprows=4)
    gdp_data = data[data['Indicator Name'] == 'GDP (current US$)']
    gdp_data = gdp_data.drop(columns=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code', 'Unnamed: 68'])
    gdp_data = gdp_data.melt(var_name='year', value_name='gdp').dropna()
    gdp_data['year'] = pd.to_numeric(gdp_data['year'], errors='coerce')

    # Generate the plot
    plt.figure(figsize=(10, 6))
    plt.scatter(gdp_data['year'], gdp_data['gdp'], color='green', label='Actual GDP')
    plt.scatter([year], [predicted_gdp], color='red', label=f'Predicted GDP for {year}')
    plt.xlabel('Year')
    plt.ylabel('GDP (Current US$)')
    plt.title(f'GDP Prediction for {year}')
    plt.legend()
    plt.grid(True)

    # Correct path to save image inside static/images/
    images_dir = os.path.join(base_dir, 'static', 'images')
    os.makedirs(images_dir, exist_ok=True)  # Make sure the folder exists

    plot_path = os.path.join(images_dir, 'plot.png')
    plt.savefig(plot_path)
    plt.close()  # Free memory

    # Return relative path to be used by Flask `url_for`
    return 'images/plot.png'
