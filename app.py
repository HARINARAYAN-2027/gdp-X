from flask import Flask, render_template, request, redirect, url_for, flash
import os
import random
from dotenv import load_dotenv
from flask_mail import Mail, Message
from visualize import generate_gdp_plot

load_dotenv()

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Mail Configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')

mail = Mail(app)

PLOT_FOLDER = os.path.join('static', 'images')
os.makedirs(PLOT_FOLDER, exist_ok=True)

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

# ✅ Results Route
@app.route('/results')
def results():
    return redirect(url_for('home'))

# ✅ GDP Prediction Route
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
            random=random.random
        )
    except Exception as e:
        return f"Error: {e}"

# ✅ Privacy Policy Route
@app.route('/privacy')
def privacy():
    return render_template('privacy.html')

# ✅ Terms & Conditions Route
@app.route('/terms')
def terms():
    return render_template('terms.html')

# ✅ Run App
if __name__ == '__main__':
    app.run(debug=True)
