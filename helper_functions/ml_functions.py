import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score



def build_model_Regression(X, model = 'RandomForestRegressor', verbose = 2, cv = 5):
    """Create pipeline with CountVectorizer, TfidfTransformer, MultioutputClassifier and RandomForestClassifier"""
    if model == 'DecisionTreeRegressor':
        parameters = {
            #RegressionTree
            'clf__max_features': [None, 20, 10],
            'clf__min_samples_leaf': [int(X.shape[0]*0.05),1000,100],
            'clf__criterion': ['mse', 'friedman_mse'],
            'clf__max_depth': [3,5,10],
        }
        pipeline = Pipeline([
            ('clf', DecisionTreeRegressor(random_state=42 ))
        #('clf', RandomForestRegressor(random_state=42 ))
        ])
    elif model == 'RandomForestRegressor':
        parameters = {
            'clf__max_features': [None, 20, 10],
            'clf__min_samples_leaf': [int(X.shape[0]*0.05),1000,100],
            'clf__criterion': ['mse', 'mae'],
            'clf__max_depth': [10, 15],
            #'clf__ccp_alpha': [0],
            'clf__n_estimators': [100, 1000],
            'clf__bootstrap': [True, False]

        }
        pipeline = Pipeline([
            ('clf', RandomForestRegressor(random_state=42 ))
        ])
    else:
        print('No model selected')
    #print(pipeline.get_params())
    cv = GridSearchCV(pipeline, parameters, return_train_score = True, verbose = verbose, scoring= 'r2', cv = cv)
    return cv

def build_model_Classification(X, model = ''):
    """Create pipeline with CountVectorizer, TfidfTransformer, MultioutputClassifier and RandomForestClassifier"""
    parameters = {
        #RegressionTree
       #'clf__criterion': ['gini','entropy'],
        'clf__max_depth': [None,10, 15], 
        #'clf__max_features': [X.shape[1],5, 10], 
        'clf__min_samples_leaf': [1,int(len(X)*0.05),1000] ,
        #'clf__min_samples_split': 2, 
        #'clf__min_weight_fraction_leaf': 0.0, 
        'clf__n_estimators': [100,10,50]

    }
    
    print('generating pipeline')
    pipeline = Pipeline([
        ('clf', RandomForestClassifier(random_state=42 ))
    ])

    #print(pipeline.get_params())
    cv = GridSearchCV(pipeline, parameters)
    return cv

def evaluate_model(model, X_test, Y_test):
    Y_pred = model.predict(X_test)
    errors = abs(Y_pred - Y_test)
    mape = 100 * np.mean(errors / Y_test)
    accuracy = 100 - mape
    r2 = r2_score(Y_test, Y_pred)
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    print('R^2 Sscore of model: ', r2_score(Y_test, Y_pred))
    
    return y_pred, accuracy, mape, r2
    

    

