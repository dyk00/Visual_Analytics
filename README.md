# Flightsaver

Flightsaver is a tool developed to help passengers book flights with minimal expected delay.

## Installation

To make sure all required libraries are installed on your system, use the following command to install the requirements:

```bash
pip install -r requirements.txt
```

Make sure that the folder `models` exists. If it does not, create a new folder called `models`.
If the folder exists, check if it contains the following files:
* arr_model.joblib
* dep_model.joblib

If the folder or these files do not exist, go to the file `app.py` and uncomment lines 37-40. This makes sure that the models are trained.

## File structure

The application has the following structure:
* `assets` folder
    * `style.css`: a stylesheet file that is part of the application layout
* `data` folder
    * `L_AIRPORT_ID.csv`: a dataset with ids and names of the airports used in the delay dataset
    * `L_UNIQUE_CARRIER.csv`: a dataset with ids and names of the carriers used in the delay dataset
    * `T_ONTIME_REPORTING.csv`: the dataset with flight delays from the year 2023, with origin Atlanta
* `models` folder
    * `arr_model.joblib`: the model used to make predictions for arrival delay
    * `dep_model.joblib`: the model used to make predictions for departure delay
* `src` folder
    * `bargraph_arr.py`: a python file with the functions to train the arrival delay model, predict values, and create a bar chart displaying the predictions
    * `bargraph_dep.py`: a python file with the functions to train the departure delay model, predict values, and create a bar chart displaying the predictions
    * `calendarheatmap.py`: a python file with the functions to create a heatmap for a delay overview per day of the year
    * `causesbarchart.py`: a python file with the functions to create a bar chart for an overview of the most common delay causes
    * `featureinfo.py`: a python file with the functions to create a bar chart for a delay overview for features carrier, origin and destination
    * `helpers.py`: a python file with several helper functions to create dropdown data
    * `modelfeatures.py`: a python file with the functions to create a bar chart for an overview of the feature importances for a given model
    * `timeheatmap.py`: a python file with the functions to create a heatmap for a delay overview per hour of the day
* `app.py`: a python file with the full layout of the application and its callbacks
* `requirements.txt`: the file used to install all necessary requirements for the application
* `README.md`

All `*.py` files are documented and contain comments for every function.

## Running the application

To run the application, use the following command:
```bash
python app.py
```