
from flask import Flask, request,jsonify
import pandas as pd
import pickle

# Create a Flask object to run
app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to IRIS classification!!"

@app.route('/predict',methods=['POST'])
def predict():
    
    rec_json = request.get_json()
    test = pd.read_json(rec_json, orient='records')
    class_predicted = int(g_IrisModel.predict(test)[0])
    
    finaloutput = pd.DataFrame({'class':[class_predicted]})
    responses = jsonify(predictions = finaloutput.to_json(orient = 'records'))
    return (responses)
	
#load the pretrained model
def load_model():
	global g_IrisModel
	
	fIrisFile = open('models/SVMModel.pckl', 'rb')
	g_IrisModel = pickle.load(fIrisFile)
	fIrisFile.close()

if __name__ == "__main__":
	print("**Starting IRIS Server")
	
	# load the Model
	load_model()
	
	app.run()