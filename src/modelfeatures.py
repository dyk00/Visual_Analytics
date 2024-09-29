import pandas as pd
import plotly.graph_objs as go


def rename_feature_cols(row):
    """
    Renames the feature column names to understandable names

    :param row: the row to rename the feature name to
    :returns: the row with the feature name renamed
    """

    if row["Feature"] == "CRS_DEP_10MIN":
        row["Feature"] = "Time"
    elif row["Feature"] == "CRS_ARR_10MIN":
        row["Feature"] = "Time"
    elif row["Feature"] == "OP_UNIQUE_CARRIER":
        row["Feature"] = "Carrier"
    elif row["Feature"] == "ORIGIN_AIRPORT_ID":
        row["Feature"] = "Starting point"
    elif row["Feature"] == "DEST_AIRPORT_ID":
        row["Feature"] = "Destination"
    elif row["Feature"] == "DAY_OF_MONTH":
        row["Feature"] = "Day of the month"
    elif row["Feature"] == "MONTH":
        row["Feature"] = "Month"

    return row


def create_feature_importance_graph(model, df):
    """
    Creates a sorted bar chart with the feature importance
    percentages

    :param model: the model to get the feature importances from
    :param df: the dataset with only the feature columns for the model
    :returns: a bar chart with the percentages of feature importance
    """

    # get the importances from the model
    importances = pd.DataFrame(
        {"Value": model.feature_importances_, "Feature": df.columns}
    )

    # get the total importance and calculate the percentages
    totalimportance = importances["Value"].sum()
    if totalimportance > 0:
        importances["Percentage"] = (importances["Value"] / totalimportance) * 100
    else:
        importances["Percentage"] = 0

    # rename the features for legibility
    importances = importances.apply(rename_feature_cols, axis=1)

    # sort the importance percentages
    importances = importances.sort_values(by="Percentage", ascending=True)

    # create the bar chart
    fig = go.Bar(
        x=importances["Percentage"],
        y=importances["Feature"],
        orientation="h",
        marker_color="rgb(0,102,255)",
        hovertemplate="Feature: %{y}<br>Importance: %{x:.1f} percent<extra></extra>",
    )

    # create the layout for the plot
    layout = go.Layout(
        height=210,
        width=300,
        margin={"t": 40},
        plot_bgcolor="white",
        title="Influence of each feature",
        font_family="Arial, Helvetica, sans-serif",
        title_x=0.5,
        yaxis_type="category",
    )

    return go.Figure(data=fig, layout=layout)
