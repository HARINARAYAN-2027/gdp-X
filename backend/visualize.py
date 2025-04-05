import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os

# Use the Agg backend for non-interactive mode
matplotlib.use('Agg')  # Disable GUI-based backends

def generate_gdp_plot(year, predicted_gdp):
    # Path to the GDP dataset
    file_path = 'backend/model/API_IND_DS2_en_csv_v2_14851/API_IND_DS2_en_csv_v2_14851.csv'
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at: {file_path}")

    # Load the dataset
    data = pd.read_csv(file_path, skiprows=4)
    gdp_data = data[data['Indicator Name'] == 'GDP (current US$)']
    gdp_data = gdp_data.drop(columns=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code', 'Unnamed: 68'])
    gdp_data = gdp_data.melt(var_name='year', value_name='gdp').dropna()
    gdp_data['year'] = pd.to_numeric(gdp_data['year'], errors='coerce')

    # Generate the plot
    plt.figure(figsize=(10, 6))
    plt.scatter(gdp_data['year'], gdp_data['gdp'], color='blue', label='Actual GDP')
    plt.scatter([year], [predicted_gdp], color='red', label=f'Predicted GDP for {year}')
    plt.xlabel('Year')
    plt.ylabel('GDP (Current US$)')
    plt.title(f'GDP Prediction for {year}')
    plt.legend()
    plt.grid(True)

    # Save the plot to the static folder
    plot_path = os.path.join('backend', 'static', 'images', 'plot.png')
    plt.savefig(plot_path)
    plt.close()  # Ensure plot is properly closed to free memory/resources

    return plot_path