import pytest
from ml.model import train_model, compute_model_metrics, inference
from ml.data import process_data
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np



# See if model returns an instance of LogisticRegression
def test_train_model():
    """
    # Training model test to validate the type of processed data
    """
    X_train = np.random.rand(100,30)
    y_train = np.random.randint(2,size=100)

    model = train_model(X_train, y_train)

    assert isinstance(model, LogisticRegression), f"Expected LogisticRegression, but got {type(model)}"


def test_compute_model_metrics():
    y_true=np.array([1,0,0,1])
    y_pred=np.array([1,0,1,1])
    
    precision, recall, fbeta = compute_model_metrics(y_true,y_pred)

    assert isinstance(precision, float),f"Expected float, but got {type(precision)}"
    assert isinstance(recall,float), f"Expected float, but got {type(recall)}"
    assert isinstance(fbeta,float), f"Expected float, but got {type(fbeta)}"
    
    

# TODO: implement the third test. Change the function name and input as needed
def test_process_data():
    data= pd.DataFrame({
    'age': [25,30,40,22],
    'workclass':['Private','Self-emp-inc', 'Private','Local-gov'],
    'education':['Bachelor', 'HS-grad', 'Masters', 'PhD'],
    'hours-per-week': [40,50,60,30],
    'salary': ['<=50K','<=50K', '>50K', '>50K']
    })
    cat_features= ['workclass', 'education']
    label='salary'

    X_processed, y_processed, encoder, lb = process_data(
        data, categorical_features=cat_features, label=label, training=True
    )
    assert X_processed.shape[1] > len(cat_features), f"Expected more features after encoding, but got {X_processed.shape[1]}"
    assert X_processed.shape[0] == len(y_processed), f"X and y should have the same number of rows, but got {X_processed.shape[0]} and {len(y_processed)}"

    