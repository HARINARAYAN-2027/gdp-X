from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file
import os
import random
import pickle
from dotenv import load_dotenv
from flask_mail import Mail, Message
from news_data import news
from news_descriptions import descriptions
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from visualize import generate_gdp_json_plot

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "default_secret_key")

# Mail configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')

mail = Mail(app)

@app.route('/')
def home():
    for item in news:
        image_name = item["image"]
        item["description"] = descriptions.get(image_name, "Description not available.")
    return render_template('home.html', news=news)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')
        msg = Message(
            subject=f"New message from {name}",
            sender=app.config['MAIL_USERNAME'],
            recipients=[app.config['MAIL_USERNAME']],
            body=f"Name: {name}\nEmail: {email}\n\nMessage:\n{message}"
        )
        try:
            mail.send(msg)
            flash('Message sent successfully!', 'success')
        except Exception as e:
            flash(f'Error sending message: {str(e)}', 'danger')
        return redirect(url_for('contact'))
    return render_template('contact.html')

@app.route('/api/plot', methods=['POST'])
def api_plot():
    data = request.get_json()
    year = int(data.get('year', 2025))
    try:
        plot_dict, predicted_gdp = generate_gdp_json_plot(year)
        return jsonify(plot=plot_dict)
    except Exception as e:
        return jsonify(error=str(e)), 500

@app.route('/privacy')
def privacy():
    return render_template('privacy.html')

@app.route('/terms')
def terms():
    return render_template('terms.html')

@app.route('/results')
def results():
    return redirect(url_for('home'))

@app.route('/predict', methods=['POST'])
def predict():
    year = int(request.form['year'])
    try:
        plot_dict, predicted_gdp = generate_gdp_json_plot(year)
        prediction_text = f"Predicted GDP for {year}: {predicted_gdp:.2f} Trillion USD"
        return render_template('result.html',
                               prediction_text=prediction_text,
                               plot_data=plot_dict['data'],
                               plot_layout=plot_dict['layout'])
    except Exception as e:
        error = f"Prediction failed: {str(e)}"
        return render_template('home.html', error=error, news=news)

@app.route('/news/<int:news_id>')
def news_detail(news_id):
    selected_news = next((item for item in news if item["id"] == news_id), None)
    if selected_news:
        selected_news["description"] = descriptions.get(selected_news["image"], "Description not available.")
        return render_template('news_detail.html', news=selected_news)
    return "News not found", 404

@app.route('/download-gdp-csv')
def download_gdp_csv():
    csv_path = os.path.join(app.root_path, 'backend', 'model', 'gdp.csv')
    return send_file(csv_path, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
