from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved model (first 1000 rows)
model = joblib.load('house_price_model_1000.pkl')

@app.route('/')
def home():
    """Render the front-end HTML form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get input data from the form
        age = float(request.form['age'])
        distance = float(request.form['distance'])
        stores = int(request.form['stores'])

        # Validate input values
        if age <= 0 or distance <= 0 or stores < 0:
            return jsonify({'price': 'No reasonable house found'})

        # Prepare data for prediction
        input_data = np.array([[age, distance, stores]])
        prediction = model.predict(input_data)[0]

        # Return the result as JSON
        return jsonify({'price': f'Predicted Price: ${prediction:.2f} per unit area'})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
