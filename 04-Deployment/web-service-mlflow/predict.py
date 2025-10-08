import pickle

from flask import Flask, request, jsonify
import mlflow.sklearn

#loading the model 
run_id="6b670d2865fe4e42bd49e04afaa6c113"
logged_model=f"runs:/{run_id}/model"
mlflow.set_tracking_uri("http://127.0.0.1:5000")

#Load model as a PyFuncModel
model=mlflow.pyfunc.load_model(logged_model)


#loading the DictVectorizer
dv_run_id="80d1a2a434b9493c84eff78c691f0700"

path=mlflow.artifacts.download_artifacts(run_id=dv_run_id, artifact_path="dict_vectorizer.bin")
print("downloading the dict vectorizer")
with open(path, 'rb') as f_out:
    dv=pickle.load(f_out)



def prepare_features(ride):
    features={}
    features['PU_DO']= '%s_%s' % (ride['PULocationID'], ride['DOLocationID'])
    features['trip_distance']=ride['trip_distance']
    return features


def predict(features):
    X=dv.transform(features)
    preds=model.predict(X)
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