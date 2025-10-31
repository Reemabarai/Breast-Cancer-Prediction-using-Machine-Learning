from flask import Flask , render_template , request
import pickle
import numpy as np
import pandas as pd
# Load the models and scaler
model = pickle.load(open("model.pkl",'rb'))

#flask app
app = Flask(__name__)


@app.route('/')
def index():
    return render_template("index.html")
    
@app.route("/predict", methods=['POST'])    
def predict():
   features=request.form['feature']
   features_lst = features.split(',')
   np_features = np.asanyarray(features_lst,dtype=np.float32)
   pred = model.predict(np_features.reshape(1,-1))
   print("Prediction Result :",pred)

   output = ["Cancerous" if pred[0] == 1 else "Not Cancerous"]

   return render_template('index.html', message = output)


#python main 
if __name__ =="__main__":
    app.run(debug=True)