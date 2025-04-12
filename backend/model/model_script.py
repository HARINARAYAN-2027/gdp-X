import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pickle
import os


file_path = 'c:/Users/harin/OneDrive/Desktop/gdp-prediction-website/backend/model/india_gdp/gdp.csv'

try:
    
    data = pd.read_csv(file_path, skiprows=4)  
    print("Dataset loaded successfully!")
    
   
    gdp_data = data[data['Indicator Name'] == 'GDP (current US$)']  
    print(f"GDP Data filtered. Shape: {gdp_data.shape}")

   
    gdp_data = gdp_data.drop(columns=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'], errors='ignore')  
    gdp_data = gdp_data.melt(id_vars=[], var_name='year', value_name='gdp')  
    gdp_data['year'] = pd.to_numeric(gdp_data['year'], errors='coerce')  
    gdp_data['gdp'] = pd.to_numeric(gdp_data['gdp'], errors='coerce')  
    gdp_data = gdp_data.dropna()  
    print("Data cleaned and reshaped successfully!")
    print(gdp_data.head()) 

  
    scaler = StandardScaler()
    features = gdp_data[['year']] 
    target = gdp_data['gdp']  
    X_scaled = scaler.fit_transform(features)  
    print("Feature scaling completed.")

    
    scaler_save_path = 'c:/Users/harin/OneDrive/Desktop/gdp-prediction-website/backend/model/scaler.pkl'
    with open(scaler_save_path, 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)
    print(f"Scaler saved successfully at: {scaler_save_path}")

   
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, target, test_size=0.2, random_state=42)
    print(f"Train-Test Split completed: {len(X_train)} training samples, {len(X_test)} testing samples.")

    
    model = RandomForestRegressor(max_depth=10, n_estimators=100, random_state=42)  
    model.fit(X_train, y_train)
    print("Model training completed.")

    
    training_accuracy = model.score(X_train, y_train)
    testing_accuracy = model.score(X_test, y_test)
    print(f"Training accuracy: {training_accuracy:.2f}")
    print(f"Testing accuracy: {testing_accuracy:.2f}")

   
    cv_scores = cross_val_score(model, X_scaled, target, cv=5, scoring='r2')  
    print(f"Cross-validation R^2 scores: {cv_scores}")
    print(f"Mean Cross-validation R^2 score: {cv_scores.mean():.2f}")

    
    model_save_path = 'c:/Users/harin/OneDrive/Desktop/gdp-prediction-website/backend/model/gdp_model.pkl'
    with open(model_save_path, 'wb') as model_file:
        pickle.dump(model, model_file)
    print(f"Model saved successfully at: {model_save_path}")

except FileNotFoundError:
    print(f"File not found! Ensure the file exists at: {file_path}")
except KeyError as e:
    print(f"KeyError: Missing required column in the dataset: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
