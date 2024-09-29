import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import plotly.express as px
import datetime
import shap


def process_data(df):
    """
    Processes the dataset and removes NaN values from columns of interest,
    Creates 10 minute time periods for all arrival times

    :param df: the dataset to process
    :returns: the processed dataset, and the feature columns to use
    """
    df = df.copy()

    # define columns of interest to train the model;
    # origin airport, destination airport, date, planned time (CRS_ARR), actual time (ARR)
    feature_columns = ["MONTH", "DAY_OF_MONTH", "CRS_ARR_TIME", "ARR_TIME"]

    # these cells of ARR_TIME have NaN values because the flights were cancelled
    # so just simply drop them
    df = df.dropna(subset=["ARR_TIME"])
    df = df.dropna(subset=["ARR_DELAY_NEW"])

    # explicitly define categorical variables, which are handled automatically in lightGBM
    categorical_columns = [
        "DEST_AIRPORT_ID",
        "ORIGIN_AIRPORT_ID",
        "OP_UNIQUE_CARRIER",
    ]
    for column in categorical_columns:
        df[column] = df[column].astype("category")
        feature_columns.append(column)

    # discretize the scheduled arrival time into hour + minute (rounded up to the nearest 10)
    df["CRS_ARR_10MIN"] = df["CRS_ARR_TIME"].apply(
        lambda x: (x // 100) * 100 + round((x % 100) / 10) * 10
    )
    feature_columns.append("CRS_ARR_10MIN")
    feature_columns.remove("CRS_ARR_TIME")
    feature_columns.remove("ARR_TIME")

    # to return and reuse df and feature columns for training the model
    return df, feature_columns


def train_model(df, feature_columns):
    """
    Trains a LightGBM model on the arrival delay

    :param df: the dataset to train on
    :param feature_columns: the columns to use for training
    :returns: the trained model, X_test, and X_train
    """
    # define features and target
    X = df[feature_columns]
    y = df["ARR_DELAY_NEW"]

    # split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # train the model using lightGBM
    model = lgb.LGBMRegressor(
        n_estimators=10000,
        learning_rate=0.01,
        random_state=42,
        force_col_wise="true",
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="l1",
        # could increase this to 50
        callbacks=[lgb.early_stopping(50)],
    )

    # To evaluate the model performance: uncomment the following code
    # # predict and evaluate using RMSE
    # y_pred = model.predict(X_test)

    # # squared=False because we want RMSE and not MSE
    # rmse = mean_squared_error(y_test, y_pred, squared=False)
    # print(f"RMSE: {rmse}")

    # # standardized rmse, divided by standard deviation of y
    # standardized_rmse = rmse / np.std(y)
    # print(f"Standardized RMSE: {standardized_rmse}")

    # print(f"Standard deviation of y: {np.std(y)}")
    # print(f"Mean of y: {np.mean(y)}")

    # return the trained model and X_test for shap
    return model, X_test, X_train


def generate_time_interval(day, month, interval):
    """
    Generates the time interval moments to predict values for

    :param day: the day selected in the input
    :param month: the month selected in the input
    :param interval: the time interval (2 or 4 days)
    :returns: a list of moments to predict delays for
    """
    base_date = datetime.datetime(2024, month, day)
    # if interval == 4 days, the start date is 1 day before, end date is 2 days after
    if interval == 4:
        start_date = base_date - datetime.timedelta(days=1)
        end_date = base_date + datetime.timedelta(days=2)
    # otherwise the end date is 1 day after
    elif interval == 2:
        start_date = base_date
        end_date = base_date + datetime.timedelta(days=1)

    # store all time ticks within the interval
    generated_times = []
    current_date = start_date
    while current_date <= end_date:
        for hour in range(24):
            for minute in range(0, 60, 10):
                generated_times.append(
                    (current_date.month, current_date.day, hour * 100 + minute)
                )
        current_date += datetime.timedelta(days=1)

    return generated_times


def predict_delays(model, day, month, carrier, origin_airport, dest_airport, interval):
    """
    Predicts arrival delays for selected date, carrier(s),
    destination, origin, and interval

    :param model: the model to use for the predictions
    :param day: the day to predict for
    :param month: the month to predict for
    :param carrier: the carrier to predict for
    :param origin_airport: the origin to predict for
    :param dest_airport: the destination to predict for
    :param interval: the interval of times to predict for
    :returns: a list of predictions for the given values
    """
    # creating new dataframe for prediction
    instance_df = pd.DataFrame(
        columns=[
            "DAY_OF_MONTH",
            "MONTH",
            "ORIGIN_AIRPORT_ID",
            "DEST_AIRPORT_ID",
            "OP_UNIQUE_CARRIER",
            "CRS_ARR_10MIN",
        ]
    )

    # generate time ticks
    times = generate_time_interval(day, month, interval)

    # predict delays for each time
    for i in range(len(times)):
        month, day, time = times[i]
        instance_df.loc[i] = {
            "DAY_OF_MONTH": day,
            "MONTH": month,
            "OP_UNIQUE_CARRIER": carrier,
            "ORIGIN_AIRPORT_ID": origin_airport,
            "DEST_AIRPORT_ID": dest_airport,
            "CRS_ARR_10MIN": time,
        }

    # have to match data types for model and input
    categorical_columns = ["ORIGIN_AIRPORT_ID", "DEST_AIRPORT_ID", "OP_UNIQUE_CARRIER"]
    for column in categorical_columns:
        if column in instance_df.columns:
            instance_df[column] = instance_df[column].astype("category")

    # predict the delays
    y_pred = model.predict(instance_df)
    instance_df["ARR_DELAY_NEW"] = y_pred

    return instance_df, y_pred


def get_feature_importance(model, X_train, feature_columns):
    """
    (Mostly) unused function: get the shap values for the model

    :param model: the model to get shap values for
    :param X_train: the training set of the model
    :param feature_columns: the features to use for the shap explainer
    :returns: the shap values for the model
    """
    tree_explainer = shap.TreeExplainer(model)
    shap_values = tree_explainer.shap_values(X_train)
    shap_df = pd.DataFrame(shap_values, columns=feature_columns)
    return shap_df


def create_bargraph(
    model, day, month, carrier1, carrier2, origin_airport, dest_airport, interval
):
    """
    Creates a bar chart with x-axis as scheduled arrival time (in 10 minute
    intervals) and y-axis as predicted delays for 2 carriers

    :param model: the model for the predictions
    :param day: the day to predict for
    :param month: the month to predict for
    :param carrier1: the first carrier to predict for
    :param carrier2: the second carrier to predict for
    :param origin_airport: the origin to predict for
    :param dest_airport: the destination to predict for
    :param interval: the time interval for which to predict
    :returns: a bar chart with the predictions
    """
    # predictions for carrier 1
    predictions1, _ = predict_delays(
        model, day, month, carrier1, origin_airport, dest_airport, interval
    )

    # predictions for carrier 2
    predictions2, _ = predict_delays(
        model, day, month, carrier2, origin_airport, dest_airport, interval
    )

    # get both in the same dataset
    merged_predictions = pd.concat([predictions1, predictions2]).reset_index(drop=True)

    # rename the columns for date
    merged_predictions = merged_predictions.rename(
        columns={"DAY_OF_MONTH": "day", "MONTH": "month"}
    )

    # get the date time moments for the x-axis
    merged_predictions["DATE_COMBINED"] = pd.to_datetime(
        merged_predictions.assign(
            year=2024,
            hour=merged_predictions["CRS_ARR_10MIN"] // 100,
            minute=round((merged_predictions["CRS_ARR_10MIN"] % 100) / 10) * 10,
        )[["year", "month", "day", "hour", "minute"]]
    )

    # the try except is for a bug in dash, to avoid a ValueError
    try:
        # create the bar chart
        bargraph = px.bar(
            merged_predictions,
            x="DATE_COMBINED",
            y="ARR_DELAY_NEW",
            color="OP_UNIQUE_CARRIER",
            color_discrete_sequence=["#0066ff", "#ffcc00"],
            labels={
                "DATE_COMBINED": "Date, Time",
            },
            custom_data=[
                "month",
                "day",
                "OP_UNIQUE_CARRIER",
                "ORIGIN_AIRPORT_ID",
                "DEST_AIRPORT_ID",
                "CRS_ARR_10MIN",
            ],
            template="plotly_white",
            barmode="group",
            hover_data={
                "OP_UNIQUE_CARRIER": True,
                "DATE_COMBINED": True,
                "ARR_DELAY_NEW": ":.3f",
            },
        )
    except:
        # create the bar chart anyway
        bargraph = px.bar(
            merged_predictions,
            x="DATE_COMBINED",
            y="ARR_DELAY_NEW",
            color="OP_UNIQUE_CARRIER",
            color_discrete_sequence=["#0066ff", "#ffcc00"],
            labels={
                "DATE_COMBINED": "Date, Time",
            },
            custom_data=[
                "month",
                "day",
                "OP_UNIQUE_CARRIER",
                "ORIGIN_AIRPORT_ID",
                "DEST_AIRPORT_ID",
                "CRS_ARR_10MIN",
            ],
            template="plotly_white",
            barmode="group",
            hover_data={
                "OP_UNIQUE_CARRIER": True,
                "DATE_COMBINED": True,
                "ARR_DELAY_NEW": ":.3f",
            },
        )

    # create the hover template for both carriers
    bargraph.data[
        0
    ].hovertemplate = "Flight date: %{x} <br>Carrier: %{customdata[2]}<br>Predicted delay: %{y:.1f} minutes"
    bargraph.data[
        1
    ].hovertemplate = "Flight date: %{x} <br>Carrier: %{customdata[2]}<br>Predicted delay: %{y:.1f} minutes"

    # create the layout of the plot
    bargraph.update_layout(
        barmode="group",
        font_family="Arial, Helvetica, sans-serif",
        yaxis_title=None,
        legend_title="Carrier",
        xaxis_tickangle=-45,
        xaxis_tickformat="%B %-d<br>%H:%M",
        xaxis_rangeslider=dict(visible=True),
        margin={
            "l": 0,
            "t": 10,
        },
        height=230,
    )

    return bargraph
