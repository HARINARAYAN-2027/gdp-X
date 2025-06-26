import pandas as pd
import numpy as np
import os
import pickle
import plotly.graph_objs as go
import matplotlib.pyplot as plt


def estimate_future_features(df, year, feature_names):
    estimated_row = {}
    for feature in feature_names:
        recent = df[['Year', feature]].dropna().sort_values(by='Year').tail(5)
        if len(recent) >= 2:
            x = recent['Year'].values
            y = recent[feature].values
            slope, intercept = np.polyfit(x, y, 1)
            estimated_value = slope * year + intercept
        else:
            estimated_value = df[feature].iloc[-1]
        estimated_row[feature] = estimated_value
    return pd.DataFrame([estimated_row])


def generate_gdp_json_plot(year):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'backend', 'model', 'gdp.csv')
    model_path = os.path.join(base_dir, 'backend', 'model', 'gdp_model.pkl')

    with open(model_path, 'rb') as f:
        model_package = pickle.load(f)
    model = model_package['model']
    feature_names = model_package['features']

    df = pd.read_csv(data_path)
    for col in df.columns:
        if col != 'Year':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['Year', 'GDP (US$ Trillion)'])
    df['Year'] = df['Year'].astype(int)

    if year in df['Year'].values:
        input_df = df[df['Year'] == year][feature_names]
    else:
        input_df = estimate_future_features(df, year, feature_names)

    predicted_log_gdp = model.predict(input_df)[0]
    predicted_gdp = np.exp(predicted_log_gdp)

    trace_actual = go.Scatter(
        x=df['Year'].astype(int).tolist(),
        y=df['GDP (US$ Trillion)'].astype(float).tolist(),
        mode='lines+markers',
        name='Actual GDP',
        line=dict(color='green')
    )

    trace_pred = go.Scatter(
        x=[year],
        y=[predicted_gdp],
        mode='markers+text',
        name='Predicted GDP',
        marker=dict(color='red', size=12),
        text=[f"{predicted_gdp:.2f} Trillion"],
        textposition='top center'
    )

    layout = go.Layout(
        title=f"GDP Prediction for {year}",
        xaxis=dict(title='Year'),
        yaxis=dict(title='GDP (US$ Trillion)'),
        hovermode='closest'
    )

    fig = go.Figure(data=[trace_actual, trace_pred], layout=layout)
    return fig.to_dict(), predicted_gdp


def generate_gdp_plot(year):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'backend', 'model', 'gdp.csv')
    model_path = os.path.join(base_dir, 'backend', 'model', 'gdp_model.pkl')

    with open(model_path, 'rb') as f:
        model_package = pickle.load(f)
    model = model_package['model']
    feature_names = model_package['features']

    df = pd.read_csv(data_path)
    for col in df.columns:
        if col != 'Year':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['Year', 'GDP (US$ Trillion)'])
    df['Year'] = df['Year'].astype(int)

    if year in df['Year'].values:
        input_df = df[df['Year'] == year][feature_names]
    else:
        input_df = estimate_future_features(df, year, feature_names)

    predicted_log_gdp = model.predict(input_df)[0]
    predicted_gdp = np.exp(predicted_log_gdp)

    
    plt.figure(figsize=(10, 6))
    plt.plot(df['Year'], df['GDP (US$ Trillion)'], marker='o', color='green', label='Actual GDP')
    plt.scatter([year], [predicted_gdp], color='red', s=100, label=f'Predicted {year}')
    plt.title(f"GDP Prediction for {year}")
    plt.xlabel("Year")
    plt.ylabel("GDP (US$ Trillion)")
    plt.legend()
    plt.grid(True)

    output_path = os.path.join('static', 'images', f'gdp_plot_{year}.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

    return predicted_gdp, output_path


if __name__ == '__main__':
    try:
        user_input = input("📅 Enter the year for GDP prediction (e.g., 2030): ")
        year_to_predict = int(user_input)
        fig, gdp = generate_gdp_json_plot(year_to_predict)
        print(f"✅ Predicted GDP for {year_to_predict}: {gdp:.2f} Trillion USD")
    except Exception as e:
        print("❌ Error:", e)
