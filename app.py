from flask import Flask, render_template, request, redirect, url_for, flash
import os
import flask_mail
import random  # Add this import
import gunicorn

# Import the function from visualize.py
from visualize import generate_gdp_plot

from flask_mail import Mail, Message  # ✅ Mail imports

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Flash messages ke liye zaroori

# ✅ Flask-Mail configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'harinarayankumar548@gmail.com'  # ✅ Replace with your Gmail
app.config['MAIL_PASSWORD'] = 'your_app_password'  # ✅ Replace with your App Password

mail = Mail(app)

# ✅ Path for saving plot
PLOT_FOLDER = os.path.join('static', 'images')
os.makedirs(PLOT_FOLDER, exist_ok=True)  # Make sure the folder exists

# ✅ Home Route
@app.route('/')
def home():
    return render_template('home.html')

# ✅ About Route
@app.route('/about')
def about():
    return render_template('about.html')

# ✅ Contact Route
@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']

        # ✅ Send Email
        msg = Message(subject=f"New message from {name}",
                      sender=app.config['MAIL_USERNAME'],
                      recipients=['harinarayankumar548@gmail.com'],  # ✅ Your receiving email
                      body=f"Name: {name}\nEmail: {email}\n\nMessage:\n{message}")
        try:
            mail.send(msg)
            flash('Message sent successfully!', 'success')
        except Exception as e:
            flash(f'Error sending message: {str(e)}', 'danger')

        return redirect(url_for('contact'))
    return render_template('contact.html')

# ✅ Results Route (optional)
@app.route('/results')
def results():
    return redirect(url_for('home'))

# ✅ Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        year = int(request.form['year'])
        predicted_gdp, plot_path = generate_gdp_plot(year)
        prediction_text = f"Predicted GDP for the year {year} is approximately {predicted_gdp:,.2f} USD."
        rel_plot_path = os.path.relpath(plot_path, start=os.path.join(os.path.dirname(__file__), 'static'))
        return render_template(
            'result.html',
            prediction_text=prediction_text,
            plot_path=rel_plot_path.replace(os.sep, "/"),
            random=random.random  # Pass random function for cache busting
        )
    except Exception as e:
        return f"Error: {e}"

# ✅ Run the app
if __name__ == '__main__':
    app.run(debug=True)
