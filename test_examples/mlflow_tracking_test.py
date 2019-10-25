# -*- coding: utf-8 -*-
"""
Created by: zhengmingsong
Created on: 10/24/19 8:40 PM
"""

import os
import mlflow
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

os.environ["MLFLOW_S3_ENDPOINT_URL"] = 'http://localhost:9000'
os.environ["AWS_ACCESS_KEY_ID"] = 'admin'
os.environ["AWS_SECRET_ACCESS_KEY"] = 'password'

mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment('demo_experiment')


def isTasty(quality):
    if quality >= 7:
        return 1
    else:
        return 0


df = pd.read_csv('test_examples/wine-quality.csv', sep=',', quotechar='"')

df['tasty'] = df['quality'].apply(isTasty)

data = df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide',
           'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']]
target = df['tasty']

data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.33, random_state=123)

for max_depth in range(3, 6):
    simpleTree = DecisionTreeClassifier(max_depth=max_depth)
    simpleTree.fit(data_train, target_train)
    y_pred = simpleTree.predict(data_test)
    with mlflow.start_run():
        mlflow.log_param('max_dept', max_depth)
        mlflow.log_metrics({
            'Accuracy': accuracy_score(target_test, y_pred),
            'Precision': precision_score(target_test, y_pred),
            'Recall': recall_score(target_test, y_pred)
        })
        mlflow.log_artifact('./test_examples/wine-quality.csv')
