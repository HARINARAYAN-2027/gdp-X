from flask import Flask, render_template, request, url_for
from flask_mail import Mail, Message
import pickle
import os
import numpy as np
from visualize import generate_gdp_plot  # Import visualization logic
from dotenv import load_dotenv  # Load .env (optional if running locally)

# Load environment variables from .env file (only for local development)
load_dotenv()

# Initialize Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')

# Configure Flask-Mail using environment variables
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.environ.get('MAIL_DEFAULT_SENDER')

mail = Mail(app)

# Load trained GDP prediction model
model_path = os.path.join(os.path.dirname(__file__), 'model', 'gdp_model.pkl')
if os.path.exists(model_path):
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    print("Model loaded successfully!")
else:
    raise FileNotFoundError(f"Model file not found at: {model_path}")

# Load scaler
scaler_path = os.path.join(os.path.dirname(__file__), 'model', 'scaler.pkl')
if os.path.exists(scaler_path):
    with open(scaler_path, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    print("Scaler loaded successfully!")
else:
    raise FileNotFoundError(f"Scaler file not found at: {scaler_path}")

# Route for Homepage
@app.route('/')
def home():
    return render_template('home.html')

# Route for About Page
@app.route('/about')
def about():
    return render_template('about.html')

# Route for Contact Page
@app.route('/contact')
def contact():
    return render_template('contact.html')

# Route to handle contact form
@app.route('/submit_contact', methods=['POST'])
def submit_contact():
    try:
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')

        if not name or not email or not message:
            return render_template('contact.html', error_message="All fields are required! Please fill out the form completely.")

        msg = Message(
            subject=f"New Contact Form Submission from {name}",
            recipients=['harinarayankumar548@gmail.com'],  # You can use a variable here if needed
            body=f"Name: {name}\nEmail: {email}\nMessage: {message}"
        )
        mail.send(msg)

        return render_template('contact.html', success_message="Thank you for reaching out! Your message has been sent successfully.")
    except Exception as e:
        return render_template('contact.html', error_message=f"An error occurred: {str(e)}")

# Route to predict GDP
@app.route('/predict', methods=['POST'])
def predict():
    try:
        year = request.form.get('year')
        if not year or not year.isdigit():
            return render_template('result.html', prediction_text="Invalid input. Please enter a numeric year.")

        year = int(year)
        scaled_year = scaler.transform([[year]])
        predicted_gdp = model.predict(scaled_year)[0]

        # Generate plot
        plot_path = generate_gdp_plot(year, predicted_gdp)

        return render_template('result.html',
                               prediction_text=f"Predicted GDP for {year}: {predicted_gdp:,.2f}",
                               plot_path=url_for('static', filename='images/plot.png'))
    except Exception as e:
        return render_template('result.html', prediction_text=f"An error occurred: {str(e)}")

# Optional Results page
@app.route('/results')
def results():
    return render_template('result.html')

# Start the Flask server
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Use PORT from Render
    app.run(debug=True, host='0.0.0.0', port=port)
