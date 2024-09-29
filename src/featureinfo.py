# plotly
import plotly.graph_objs as go

# other libraries
import pandas as pd


def create_feature_overview(df, feature, cs, ids, locs=None, carrs=None):
    """
    Creates a bar chart with an overview for the selected feature

    :param df: the dataset to use
    :param feature: the feature to display
    :param cs: the dataset with carrier names
    :param ids: the dataset with location names
    :param locs: the selected locations
    :param carrs: the selected carriers
    :returns: a sorted bar chart with the average delay for the
              selected feature, highlighting the selected options
    """

    # if the feature is the carrier, we need to use the carrier dataset
    if feature == "OP_UNIQUE_CARRIER":
        # get the average delay for this feature
        x, y1 = get_average_delay(df, feature)

        # get the colors to use for each bar
        colors = ["rgb(0,102,255)"] * len(x)

        # make sure the name of the airline is used
        y = [cs.loc[cs["Code"] == feat]["Description"].iloc[0] for feat in y1]
        y = [(name.rsplit(" ", 1)[0] if name[-1] == "." else name) for name in y]

        # if there is a selected carrier, make sure the information is complete
        if carrs is not None:
            if carrs[0] is not None and carrs[1] is not None:
                # highlight the carriers
                colors = [
                    ("rgb(255,204,0)" if car == carrs[0] or car == carrs[1] else col)
                    for car, col in zip(y1, colors)
                ]
                pass

    # if the feature is a location one, we use the locations dataset
    else:
        # get the average delay for this feature
        x, y1 = get_average_delay(df, feature)

        # get the colors to use for each bar
        colors = ["rgb(0,102,255)"] * len(x)

        # make sure the name of the location is used
        y = [ids.loc[ids["Code"] == feat]["Description"].iloc[0] for feat in y1]
        y = [name.split(":")[1] for name in y]

        # if there is a selected location, make sure the information is complete
        if locs is not None:
            # starting point selected as feature, so highlight starting point
            if feature == "ORIGIN_AIRPORT_ID" and locs[0] is not None:
                colors = [
                    ("rgb(255,204,0)" if loc == locs[0] else col)
                    for loc, col in zip(y1, colors)
                ]
            # destination selected as feature, so highlight starting point
            elif feature == "DEST_AIRPORT_ID" and locs[1] is not None:
                colors = [
                    ("rgb(255,204,0)" if loc == locs[1] else col)
                    for loc, col in zip(y1, colors)
                ]

    # get the names for each feature to display in the title
    featuremap = {
        "OP_UNIQUE_CARRIER": "carrier",
        "ORIGIN_AIRPORT_ID": "starting point",
        "DEST_AIRPORT_ID": "destination",
    }

    hovertext = [
        airline + ": " + str(round(delay, 1)) + " minutes"
        for airline, delay in zip(y, x)
    ]

    # create a horizontal bar chart
    fig = go.Bar(
        x=x, y=y, orientation="h", marker_color=colors, name="", hovertemplate=hovertext
    )

    # create the layout for this feature
    layout = go.Layout(
        width=350,
        height=300,
        font_family="Arial, Helvetica, sans-serif",
        title="Average delay per " + featuremap[feature] + " (mins)",
        title_x=0.5,
        title_y=0.95,
        margin={"t": 40, "l": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return go.Figure(data=fig, layout=layout)


def get_average_delay(df, feature):
    """
    Returns a sorted list of ids of the selected feature
    and their average delay in the dataset

    :param df: the dataset to use
    :param feature: the feature to display values for
    :returns: a list of ids and a list of average delays
    """
    # get options for the feature
    feats = list(df[feature].unique())

    # get their average delay
    x = [df.loc[df[feature] == feat]["DEP_DELAY_NEW"].mean() for feat in feats]

    # sort both lists
    y = [feat for _, feat in sorted(zip(x, feats))]
    x = sorted(x)

    # if there are more than 10 options, show only the top 10
    if len(x) > 10:
        return x[:10], y[:10]
    else:
        return x, y
