
from flask import Flask,request, render_template
import numpy as np
import pickle


#loading models
RF = pickle.load(open('RF.pkl','rb'))
preprocessor = pickle.load(open('preprocessor.pkl','rb'))
item_list = ['Potatoes','Maize','Wheat','Rice, paddy','Soybeans','Sorghum','Sweet potatoes','Cassava','Yams',
             'Plantains and others' ]


#flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predict",methods=['POST'])
def predict():
    max_yield = 0
    best_item = None
    if request. method == 'POST':
        # Item  = request.form['Item']
        average_rain_fall_mm_per_year = request.form['average_rain_fall_mm_per_year']
        pesticides_tonnes = request.form['pesticides_tonnes']
        avg_temp = request.form['avg_temp']

        for Item in item_list:
            features = np.array([[Item,average_rain_fall_mm_per_year,pesticides_tonnes,avg_temp]],dtype=object)
            transformed_features = preprocessor.transform(features)
            predicted_value = RF.predict(transformed_features).reshape(1,-1)
            if(predicted_value > max_yield):
                max_yield = predicted_value
                best_item = Item

        return render_template('index.html',predicted_value =max_yield,best = best_item)

if __name__=="__main__":
    app.run(debug=True)