# plotly
import plotly.graph_objs as go
from plotly.express.colors import make_colorscale

# other libraries
import pandas as pd
import numpy as np


def create_clock_heatmap(df):
    """
    Creates a barpolar plotly figure for the average delay per
    hour of the day

    :param df: the dataset to use
    :returns: a barpolar figure with the average delay
    """
    # get the lists for the times of day
    am_pm_times = [1,2,3,4,5,6,7,8,9,10,11,12,
                   1,2,3,4,5,6,7,8,9,10,11,12]
    times = [1,2,3,4,5,6,7,8,9,10,11,12,
             13,14,15,16,17,18,19,20,21,22,23,0]
    # get whether a time is A.M. or P.M.
    am_pm = ["A.M.","A.M.","A.M.","A.M.","A.M.","A.M.",
             "A.M.","A.M.","A.M.","A.M.","A.M.","P.M.",
             "P.M.","P.M.","P.M.","P.M.","P.M.","P.M.",
             "P.M.","P.M.","P.M.","P.M.","P.M.","A.M."]

    # get the degree positions for each hour of the day
    th = []
    for i in times:
        if i >= 7:
            diff = i - 7
            th.append(345 - (15 * diff))
        else:
            diff = 6 - i
            th.append(0 + 15 * diff)

    # average delay is mean of dep_delay_new of every column with that specific hour
    totaldelay = []
    df["DEP_HOUR"] = np.floor(df["DEP_TIME"].div(100))
    for t in times:
        val = df.loc[df["DEP_HOUR"] == t]["DEP_DELAY_NEW"].mean()
        if not np.isnan(val):
            totaldelay.append(val)
        else:
            totaldelay.append(0)

    # get the color scale
    colmap1 = make_colorscale(["rgb(0,102,255)", "rgb(255,204,0)"])

    # create a barpolar figure: everything should be in position 1 on the r-axis
    fig = go.Barpolar(
        r=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        theta=th,
        dtheta=1,
        theta0=0,
        name="",
        marker_color=totaldelay,
        hovertemplate=[
            str(i) + " " + a + ": " + str(round(dl, 1)) + " minutes"
            for i, a, dl in zip(am_pm_times, am_pm, totaldelay)
        ],
        marker_colorscale=colmap1,
        marker_colorbar_thickness=24,
    )

    # create the layout for the figure
    layout = go.Layout(
        title="Average delay <br> per time of day (mins)",
        title_x=0.5,
        title_y=0.9,
        height=240,
        width=240,
        font_family="Arial, Helvetica, sans-serif",
        polar_bargap=0.05,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        polar={
            "radialaxis": {
                "tickmode": "array",
                "ticktext": ["1", "2"],
                "tickvals": [1, 2],
                "maxallowed": 2,
                "autorange": "reversed",
                "showline": False,
                "showticklabels": False,
            },
            "angularaxis": {
                "tickmode": "array",
                "ticktext": ["6","5","4","3","2","1",
                             "24","23","22","21","20","19",
                             "18","17","16","15","14","13",
                             "12","11","10","9","8","7"],
                "tickvals": [0,15,30,45,60,75,
                             90,105,120,135,150,165,
                             180,195,210,225,240,255,
                             270,285,300,315,330,345],
            },
        },
        margin={"t": 70, "b": 50, "l": 20, "r": 20},
    )

    # return the barpolar figure as a plotly figure
    return go.Figure(data=fig, layout=layout)
