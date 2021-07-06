import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score


def build_model_Regression(X, model = 'RandomForestRegressor'):
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
            'clf__criterion': ['mse', 'friedman_mse'],
            'clf__max_depth': [3,5,10],
            'clf__ccp_alpha': [0],

        }
        pipeline = Pipeline([
            ('clf', RandomForestRegressor(random_state=42 ))
        ])
    else:
        print('No model selected')
    #print(pipeline.get_params())
    cv = GridSearchCV(pipeline, parameters)
    return cv

def build_model_Classification(X, model = ''):
    """Create pipeline with CountVectorizer, TfidfTransformer, MultioutputClassifier and RandomForestClassifier"""
    parameters = {
        #RegressionTree
       #'clf__criterion': ['gini','entropy'],
        'clf__max_depth': [None,3,10], 
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
    print('R^2 Sscore of model: ', r2_score(Y_test, Y_pred))

    
    return Y_pred  
