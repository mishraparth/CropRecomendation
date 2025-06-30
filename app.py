from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model and scalers
model = pickle.load(open('model.pkl', 'rb'))
MS = pickle.load(open('MS.pkl', 'rb'))
SS = pickle.load(open('SS.pkl', 'rb'))

crop_type_mapping ={
    1:'rice',
    2:'maize',
    3:'chickpea',
    4:'kidneybeans',
    5:'pigeonpeas',
    6:'mothbeans',
    7:'mungbean',
    8:'blackgram',
    9:'lentil',
   10:'pomegranate',
    11:'banana',
    12:'mango',
    13:'grapes',
    14:'watermelon',
    15:'muskmelon',
    16:'apple',
    17:'orange',
    18:'papaya',
    19:'coconut',
    20:'cotton',
    21:'jute',
    22:'coffee'

}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form values
    N = float(request.form['N'])
    P = float(request.form['P'])
    K = float(request.form['K'])
    temp = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    ph = float(request.form['ph'])
    rainfall = float(request.form['rainfall'])

    input_data = np.array([[N, P, K, temp, humidity, ph, rainfall]])
    input_data = MS.transform(input_data)
    input_data = SS.transform(input_data)

    prediction = model.predict(input_data)[0]
    return render_template('index.html', result=f"🌱 Recommended Crop: {crop_type_mapping[prediction]}")

if __name__ == '__main__':
    app.run(debug=True)

