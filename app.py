from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open("model.pkl", 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_electric_consumption():
    try:
        # Extract input data
        Global_active_power = float(request.form.get('Global_active_power'))
        Global_reactive_power = float(request.form.get('Global_reactive_power'))
        Voltage = float(request.form.get('Voltage'))
        Global_intensity = float(request.form.get('Global_intensity'))
        Total_metering = float(request.form.get('Total_metering'))

        # Make prediction
        input_features = np.array([Global_active_power, Global_reactive_power, Voltage, Global_intensity, Total_metering]).reshape(1, -1)
        result = model.predict(input_features)

        return render_template('output.html', prediction_result=result[0])
    except ValueError as e:
        return "Error: Invalid input data. Please make sure all fields are filled with valid numbers."

if __name__ == '__main__':
    app.run(debug=True)
