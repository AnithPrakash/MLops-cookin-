import pandas as pd 
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
import mlflow 
import xgboost as xgb 

from pathlib import Path


mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("nyc_taxi_experiment")


models_folder = Path('models')
models_folder.mkdir(exist_ok=True)


def read_dataframe(year, month):
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet'
    df = pd.read_parquet(url)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']

    return df


def create_X(df, dv=None):
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')

    if dv is None:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    return X, dv



def train_model(X_train, y_train, X_val, y_val, dv):
    with mlflow.start_run() as run:
        train=xgb.DMatrix(X_train, label=y_train)
        valid=xgb.DMatrix(X_val, label=y_val)
        
        best_params={
        "learning_rate": 0.14988312150619953,
        "max_depth":66,
        "min_child_weight":1.0531411801474737,
        "objective":"reg:linear",
        "reg_alpha":0.01980913796072851,
        "reg_lambda":0.042524169343261004,
        "seed":13}
        
        mlflow.log_params(best_params)
        
        booster=xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=10,
            evals=[(valid, "validation")],
            early_stopping_rounds=50
        )
        
        y_pred=booster.predict(valid)
        rmse=mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse",rmse)
        
        with open("models/preprocessor.b", 'wb') as f_out:
            pickle.dump(dv, f_out)
        #mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")
        
        mlflow.xgboost.log_model(booster, name="model_mlflow", code_paths=["models"])




def run(year, month):
    df_train=read_dataframe(year=year, month=month)
    next_year=year if month < 12 else year + 1
    next_month = month + 1 if month < 12 else 1
    df_val=read_dataframe(year=next_year, month=next_month)

    X_train, dv= create_X(df_train)
    X_val, _=create_X(df_val, dv)

    target='duration'
    y_train= df_train[target].values
    y_val=df_val[target].values

    train_model(X_train, y_train, X_val, y_val, dv)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train a model to predict taxi trip duration.')
    parser.add_argument('--year', type=int, required=True, help='Year of the data to train on')
    parser.add_argument('--month', type=int, required=True, help='Month of the data to train on')
    args = parser.parse_args()

    run_id = run(year=args.year, month=args.month)

    with open("run_id.txt", "w") as f:
        f.write(run_id)
