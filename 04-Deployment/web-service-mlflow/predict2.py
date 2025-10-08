import pickle
import mlflow
import os 

from flask import Flask, request, jsonify
import mlflow.sklearn

"""Running pipeline without the mlflow tracking server
 NOT WORKING TRYING TO SOLVE THE ISSUE
 """

#loading the model 
run_id=os.getenv('RUN_ID')

#loading the model using model location 
logged_model=f"s3://s3-bucket-default-mlflow/2/{run_id}/artifacts/model"

#Load model as a PyFuncModel
model=mlflow.pyfunc.load_model(logged_model)


def prepare_features(ride):
    features={}
    features['PU_DO']= '%s_%s' % (ride['PULocationID'], ride['DOLocationID'])
    features['trip_distance']=ride['trip_distance']
    return features


def predict(features):
    preds=model.predict(features)
    return float(preds[0])

app=Flask('Duration_prediction')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    ride= request.get_json()

    features= prepare_features(ride)
    pred=predict(features)

    result={
        'duration':pred
    }

    return jsonify(result)

if __name__=="__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)