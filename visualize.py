import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os

# save graph without showing it
matplotlib.use('Agg') 

def generate_gdp_plot(year, predicted_gdp):
   
    file_path = 'model/india_gdp/gdp.csv'
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at: {file_path}")

   
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

    # Save the plot to the static folder
    plot_path = os.path.join('static', 'images', 'plot.png')
    plt.savefig(plot_path)
    plt.close()  # Close the plot to free memory
    plt.show()  

    return plot_path