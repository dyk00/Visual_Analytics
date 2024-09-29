# plotly
import plotly.graph_objs as go

# other libraries
import pandas as pd
import numpy as np

# causes and their ids
causes = {
    "Carrier": 0,
    "Weather": 1,
    "National Aviation System": 2,
    "Security": 3,
    "Late Aircraft": 4,
}


def get_barchart_data(df):
    """
    Finds the percentages of delay for all different delay
    causes

    :param df: the dataset to use
    :returns: a sorted list of delay percentages and a
              list of delay causes
    """
    # get the different cause names
    y = ["Carrier", "Weather", "National Aviation System", "Security", "Late Aircraft"]

    # get the columns to use the data from
    df_cols = df[
        [
            "CARRIER_DELAY",
            "WEATHER_DELAY",
            "NAS_DELAY",
            "SECURITY_DELAY",
            "LATE_AIRCRAFT_DELAY",
        ]
    ]

    # get the total delay for each column
    x_temp = []
    for _, col in df_cols.items():
        x_temp.append(int(np.sum(col)))

    # get the percentage by dividing by the total delay
    total_delay = np.sum(x_temp)
    if total_delay > 0:
        x = [(i / total_delay) * 100 for i in x_temp]
    else:
        x = x_temp

    # sort the lists on large percentage
    x = sorted(x)
    y = [i for _, i in sorted(zip(x, y))]

    return x, y


def create_causes_chart(df):
    """
    Creates a bar chart with the percentage of delay
    each delay cause attributes to the total delay

    :param df: the dataset to use
    :returns: a bar chart with the delay causes
    """

    # get the delay percentages and colors for each bar
    x, y = get_barchart_data(df)
    colors = ["rgb(0,102,255)"] * len(y)

    # create the horizontal bar chart
    fig = go.Bar(
        x=x,
        y=y,
        orientation="h",
        marker_color=colors,
        name="",
        hovertemplate="<b>%{y}</b><br>%{x:.1f} percent of all delay",
    )

    # create the layout
    layout = go.Layout(
        title="Most frequent causes of delay",
        title_y=1,
        title_x=0.5,
        font_family="Arial, Helvetica, sans-serif",
        margin={"t": 35, "r": 20},
        width=300,
        height=200,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return go.Figure(data=fig, layout=layout)
