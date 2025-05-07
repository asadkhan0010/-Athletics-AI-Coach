from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the saved model and scaler using pickle
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form
        data = request.form.to_dict()

        # Convert to DataFrame for model prediction
        features = pd.DataFrame([data])
        
        # Apply scaling to the input features
        features_scaled = scaler.transform(features)

        # Get prediction from the model
        prediction = model.predict(features_scaled)

        # Return the prediction
        return jsonify({'prediction': prediction[0]})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
