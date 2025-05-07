from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load your trained machine learning model
# Example: model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('/templates/index.html')  # This will render your HTML form

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    target_pace = float(request.form['Target_Pace'])
    actual_pace = float(request.form['Actual_Pace'])
    target_distance = float(request.form['Target_Distance'])
    actual_distance = float(request.form['Actual_Distance'])
    elevation_gain = float(request.form['Elevation_Gain'])
    consistency_score = float(request.form['Consistency_Score'])

    # Prepare the data in the format that your model expects
    input_data = [[target_pace, actual_pace, target_distance, actual_distance, elevation_gain, consistency_score]]
    
    # Use the model to predict
    prediction = model.predict(input_data)  # Replace with your model's predict function
    
    # Render the result page and display the prediction
    return render_template('/templates/result.html', prediction=prediction[0])

if __name__ == "__main__":
    app.run(debug=True)
