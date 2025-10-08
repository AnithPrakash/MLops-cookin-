import pickle
import mlflow

from flask import Flask, request, jsonify
import mlflow.sklearn

"""create a pipeline for the mlflow"""

#loading the model 
run_id="18a0d886c5c5494abae629e03532c2fb"
#logged_model=f"runs:/{run_id}/model"
#mlflow.set_tracking_uri("http://127.0.0.1:5000")

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