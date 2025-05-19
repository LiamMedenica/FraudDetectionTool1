import pandas as pd             #used for handling tabular data (CSV FIles)
import numpy as np              #Is a numerical computing library used to calculate mean here.
from sklearn.preprocessing import StandardScaler

def clean_data(df): #Defines a function called clean_data, takes (df) as input

    label_col = None
    for col in ['Class', 'Fraud', 'Label',  'IsFraud']:   #Looks for the label in the CSV file, (Supervised Only) 'isFraud',
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):    #Makes sure the label is numeric and not yes/no 
            label_col = col
            break
   
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]    #Looks for Columns with a numeric data type
    df_numeric = df[numeric_cols].copy()                        #creates a new dataframe with only the numeric columns
    df_numeric.fillna(df_numeric.mean(), inplace=True)     #fills the empty values with the mean    
    df_numeric.dropna(how="all", inplace=True)  #removes rows that are empty, no values at all
    
    scaler = StandardScaler()       #Creates an Instance of StandardScaler
    feature_cols = [col for col in df_numeric.columns if col != label_col]      #creates a list of column names excluding the label for scaling
    if feature_cols:
        df_numeric[feature_cols] = scaler.fit_transform(df_numeric[feature_cols])   #scales the feature columns (everything but label)

    if label_col in df_numeric.columns:              #checks the label column wasnt dropped anywhere    
        df_numeric[label_col] = df_numeric[label_col].astype(int)   #converts the label column to integer (either 1 or 0)
    return df_numeric, label_col        #returns the cleaned data frame and the label column name
