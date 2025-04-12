from flask import Flask, render_template, request, url_for, send_from_directory  # <-- Updated line
from flask_mail import Mail, Message
import pickle
import os
import numpy as np
from visualize import generate_gdp_plot


if os.path.exists("/etc/secrets/.env"):
    from dotenv import load_dotenv
    load_dotenv("/etc/secrets/.env")

app = Flask(__name__, static_folder='static', template_folder='templates')


app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_DEFAULT_SENDER')

mail = Mail(app)


model_path = os.path.join(os.path.dirname(__file__), 'model', 'gdp_model.pkl')
scaler_path = os.path.join(os.path.dirname(__file__), 'model', 'scaler.pkl')

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/submit_contact', methods=['POST'])
def submit_contact():
    try:
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')

        if not name or not email or not message:
            return render_template('contact.html', error_message="Please fill out all fields.")

        msg = Message(
            subject=f"Contact from {name}",
            recipients=['harinarayankumar548@gmail.com'],
            body=f"Name: {name}\nEmail: {email}\nMessage: {message}"
        )
        mail.send(msg)

        return render_template('contact.html', success_message="Message sent successfully.")
    except Exception as e:
        return render_template('contact.html', error_message=f"Error: {str(e)}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        year = request.form.get('year')
        if not year or not year.isdigit():
            return render_template('result.html', prediction_text="Invalid input.")

        year = int(year)
        scaled_year = scaler.transform([[year]])
        prediction = model.predict(scaled_year)[0]

        plot_path = generate_gdp_plot(year, prediction)

        return render_template(
            'result.html',
            prediction_text=f"GDP for {year}: {prediction:,.2f}",
            plot_path=url_for('static', filename='images/plot.png')
        )
    except Exception as e:
        return render_template('result.html', prediction_text=f"Error: {str(e)}")

@app.route('/results')
def results():
    return render_template('result.html')


@app.route('/privacy')
def privacy():
    return render_template('privacy.html')

@app.route('/terms')
def terms():
    return render_template('terms.html')


@app.route('/ads.txt')
def ads_txt():
    return send_from_directory('.', 'ads.txt')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
