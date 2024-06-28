from flask import Flask, render_template, request, jsonify
import main

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('ForecastIQ.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    total_price = float(request.form['total_price'])
    base_price = float(request.form['base_price'])
    units_sold = int(request.form['units_sold'])

    # Process input data using your machine learning model
    prediction = main.predict_demand(total_price, base_price, units_sold)

    # Determine prediction text based on prediction result
    if prediction == 2:
        prediction_text = "Demand for the product is High"
    elif prediction == 1:
        prediction_text = "Demand for the Product is Average"
    else:
        prediction_text = "Demand for the product is Low"

    # Return prediction result as JSON
    return jsonify({'prediction': prediction_text})

if __name__ == '__main__':
    app.run(debug=True)
