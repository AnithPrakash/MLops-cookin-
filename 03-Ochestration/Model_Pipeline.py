#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('python -V')


# In[4]:


import pandas as pd 
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error


# In[6]:


import mlflow 

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("nyc_taxi_experiment")


# In[10]:


# making it a function for easy access
def read_dataframe(filename):
    df=pd.read_parquet(filename)

    df['duration']=df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df=df[(df.duration >= 1.0) & (df.duration <= 62)]

    categorical=['PULocationID','DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    df["PU_OD"]=df["PULocationID"] + '_' + df['DOLocationID']
    
    return df


# In[16]:


df_train=read_dataframe("https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2021-01.parquet")
df_val=read_dataframe("https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2021-02.parquet")


# In[17]:


categorical=["PU_OD"]#'PULocationID','DOLocationID']
numerical=['trip_distance']

dv=DictVectorizer()

train_dicts=df_train[categorical + numerical].to_dict(orient="records")
X_train=dv.fit_transform(train_dicts)

val_dicts=df_val[categorical + numerical].to_dict(orient="records")
X_val=dv.transform(val_dicts)


# In[18]:


target="duration"
y_train=df_train[target].values
y_val=df_val[target].values


# In[11]:


# creating model on xgboost


# In[21]:


import xgboost as xgb 

from pathlib import Path


# In[22]:


models_folder = Path('models')
models_folder.mkdir(exist_ok=True)


# In[27]:


with mlflow.start_run():
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


# In[ ]:




