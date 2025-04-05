from flask import Flask, render_template, request, url_for
from flask_mail import Mail, Message
import pickle
import os
import numpy as np
from visualize import generate_gdp_plot  # Import visualization logic

# Initialize Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')

# Configure Flask-Mail
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'your-email@gmail.com'  # Replace with your Gmail address
app.config['MAIL_PASSWORD'] = 'your-app-password'    # Replace with your Gmail app password
app.config['MAIL_DEFAULT_SENDER'] = 'your-email@gmail.com'  # Replace with your Gmail address

mail = Mail(app)

# Load the trained GDP prediction model
model_path = 'backend/model/gdp_model.pkl'
if os.path.exists(model_path):
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    print("Model loaded successfully!")
else:
    raise FileNotFoundError(f"The model file was not found at: {model_path}")

# Load the scaler
scaler_path = 'backend/model/scaler.pkl'
if os.path.exists(scaler_path):
    with open(scaler_path, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    print("Scaler loaded successfully!")
else:
    raise FileNotFoundError(f"The scaler file was not found at: {scaler_path}")

# Route for Homepage
@app.route('/')
def home():
    return render_template('index.html')  # Render homepage

# Route for About Page
@app.route('/about')
def about():
    return render_template('about.html')  # Render about page

# Route for Contact Page
@app.route('/contact')
def contact():
    return render_template('contact.html')  # Render contact page

# Route for handling Contact Form submission
@app.route('/submit_contact', methods=['POST'])
def submit_contact():
    try:
        # Retrieve form data
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')

        # Validate input fields
        if not name or not email or not message:
            return render_template(
                'contact.html',
                error_message="All fields are required! Please fill out the form completely."
            )

        # Send email
        msg = Message(
            subject=f"New Contact Form Submission from {name}",
            recipients=['harinarayankumar548@gmail.com'],  # Your email address
            body=f"Name: {name}\nEmail: {email}\nMessage: {message}"
        )
        mail.send(msg)

        # Show success message
        return render_template(
            'contact.html',
            success_message="Thank you for reaching out! Your message has been sent successfully."
        )
    except Exception as e:
        return render_template(
            'contact.html',
            error_message=f"An error occurred: {str(e)}"
        )

# Route for GDP Prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input
        year = request.form.get('year')
        if not year or not year.isdigit():
            return render_template('result.html', prediction_text="Invalid input. Please enter a numeric year.")

        year = int(year)

        # Scale the input year
        scaled_year = scaler.transform([[year]])

        # Predict GDP using the trained model
        predicted_gdp = model.predict(scaled_year)[0]

        # Generate the plot dynamically
        plot_path = generate_gdp_plot(year, predicted_gdp)

        # Render the result page
        return render_template(
            'result.html',
            prediction_text=f"Predicted GDP for {year}: {predicted_gdp:,.2f}",
            plot_path=url_for('static', filename='images/plot.png')
        )
    except Exception as e:
        # Handle errors gracefully
        return render_template('result.html', prediction_text=f"An error occurred: {str(e)}")

# Route for Results Page (optional placeholder)
@app.route('/results')
def results():
    return render_template('result.html')  # Placeholder route to render results page

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)