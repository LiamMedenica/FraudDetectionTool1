from flask import Flask, render_template, request, send_file        #flask functions, handling files and web routing
import pandas as pd
from preprocess import clean_data                   #Calls the function to preprocess the data, from preprocess.py  
from model import train_and_predict                 #calls the function to train and predict, from model.py
import os
import time         #os and time, used for file naming

app = Flask(__name__)       #creates the flask web application

MAX_ROWS = 10000        #amount of rows the fraud detection tool will analyse

BASE_DIR = os.path.abspath(os.path.dirname(__file__))   #directory where script is located
RESULTS_DIR = os.path.join(BASE_DIR, "..", "results")   #path to results folder, one level above the scripts

@app.route("/", methods=["GET", "POST"])
def index():                                        #defines the endpoint that handles file uploads (POST), and default page render (GET)
    if request.method == "POST":
        file = request.files["csv_file"]            #checks for for a uploaded CSV file 
        if file:
            data = pd.read_csv(file)                #reads the uploaded CSV file into a pandas dataframe
            if len(data) > MAX_ROWS:
                data = data.head(MAX_ROWS)                                                  
                print(f"Dataset too large ({len(data)} rows), capped at {MAX_ROWS}")        #caps the dataset at whatever MAXROWS is set at
            cleaned_data, label_col = clean_data(data.copy())                               #cleans the data in proprocess.py
            X = cleaned_data.drop(columns=[label_col], errors="ignore") if label_col else cleaned_data
            y = cleaned_data[label_col] if label_col else None                                              #x = all of the columns apart from label, y = label column if any
            predictions, probs, report = train_and_predict(X, y, label_col)                                 #returns results
            if predictions is not None:
                cleaned_data["Fraud_Prediction"] = predictions
                cleaned_data["Fraud_Probability"] = probs                                            #adds predictions and probability columns to the dataframe
                fraud_cases = cleaned_data[cleaned_data["Fraud_Prediction"] == 1]
                fraud_cases = fraud_cases.sort_values(by="Fraud_Probability", ascending=False)
                fraud_count = len(fraud_cases)                                                      #filters the dataset to only include 'frauds', and sorts it by probability
                timestamp = int(time.time())
                output_path = os.path.join(RESULTS_DIR, f"fraud_cases_{timestamp}.csv")
                os.makedirs(RESULTS_DIR, exist_ok=True)  
                fraud_cases.to_csv(output_path, index=False)                                    #generates a unique filename using time, and writes all of the fraud cases to a CSV file
                if not os.path.exists(output_path):  
                    print(f"Error: File not saved at {output_path}")                       #checks the file was written
                app.config['latest_fraud_file'] = output_path                              #stores the latest output file in flask config 
                return render_template("index.html", fraud_cases=fraud_cases.to_html(), 
                                       metrics=report, file_loaded=True, fraud_count=fraud_count)           #renders the web page with, table of frauds, performance metrics, fraud count
    return render_template("index.html", fraud_cases=None, metrics=None, file_loaded=False, fraud_count=0)  #handles the page load get method, no file uploaded so no data to show 

@app.route("/download")
def download():                                     #define endpoint to download most recent file
    latest_file = app.config.get('latest_fraud_file', os.path.join(RESULTS_DIR, "fraud_cases.csv")) #fetches the path to the latest fraud results file
    if not os.path.exists(latest_file):
        print(f"Error: Download file not found at {latest_file}")
        return "File not found", 404                                        #if the file does not exist, show an error 
    return send_file(latest_file, as_attachment=True)       #sends the file to the user as a downloadable CSV file

if __name__ == "__main__":
    app.run(debug=True)             #runs the flask app in debug mode if asked 
