import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest    #Importing the models (randomforest for supervised, isolation forest for unsupervised)
from sklearn.metrics import classification_report               #generated a report which includes precision, accuracy, recall and f1 score
from sklearn.model_selection import train_test_split

def train_and_predict(X, y=None, label_col=None):
    if y is not None:                                   #checks if there is a label 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)   #uses 20% of the data for testing
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")      #uses 100 decision trees, ensures consistant results, adjusts weight to handle imbalences data
        model.fit(X_train, y_train)                         #trains the data using the 20%
        predictions = model.predict(X)                      #makes fraud preditctions (0 or 1) for the dataset
        probs = model.predict_proba(X)                      #created a probability of fraud 
        probs = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]      #
        test_preds = model.predict(X_test)                              #predicts fraud labels for the test set 
        report = classification_report(y_test, test_preds, output_dict=True, zero_division=1)           #makes a classification report 
        print(f"\nModel Performance on Test Set (Label: {label_col or 'Unknown'}):")            #Print line saying performance on (label if there is one)
        print(classification_report(y_test, test_preds))        #prints the classification report in readable text
        return predictions, probs, report       #returns everything
    else:
        print("\nWarning: No label column found (tried 'Class', 'Fraud', 'Label', 'IsFraud', 'isFraud'). Predicting anomalies unsupervised with Isolation Forest.") #print to show label was not found
        model = IsolationForest(contamination=0.002, random_state=42)  # assumes 0.2% of the data is fraud
        model.fit(X)                            #trains the model on the entire dataset
        predictions = (model.predict(X) == -1).astype(int)  # returns a score -1 for anomalies 1 for normal and flips the score
        probs = -model.decision_function(X)  # returns score negative for anomalies
        probs = (probs - probs.min()) / (probs.max() - probs.min())  # Normalize to 0-1
        return predictions, probs, None     #returns the predictions, normalised probabilities and none
    
