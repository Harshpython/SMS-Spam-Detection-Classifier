# importing flask and pickle module

from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)

# Load the trained model and vectorizer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)# loading the module
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)


# Define a route for the homepage
@app.route('/')
def home():
    return render_template('index.html')


# Define a route for predicting email spam
@app.route('/predict', methods=['POST'])
def predict_spam():
    # Get the email text from the request
    email_text = request.form['email_text']

    # Convert the text into numerical features using the trained vectorizer
    X_test = vectorizer.transform([email_text])# training the dataset

    # Predict the spam probability using the trained model
    spam_probability = model.predict_proba(X_test)[:, 1].item()

    # Return the predicted spam probability
    return render_template('index.html', spam_probability=spam_probability, email_text=email_text)


if __name__ == '__main__':
    app.run(debug=True)
