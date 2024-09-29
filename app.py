# dash
from dash import Dash, html, dcc, ctx, callback, Output, Input, State, no_update
import dash_bootstrap_components as dbc

# plotly
import plotly.graph_objs as go

# other libraries
from pathlib import Path
import pandas as pd
import datetime
import joblib

# modules
from src.calendarheatmap import create_heatmap
from src.timeheatmap import create_clock_heatmap
from src.causesbarchart import create_causes_chart
from src.featureinfo import create_feature_overview
from src.modelfeatures import create_feature_importance_graph
import src.helpers as helpers
import src.bargraph_dep as bd
import src.bargraph_arr as ba

# read in the data
df = pd.read_csv(
    str(Path.cwd()) + "\data\T_ONTIME_REPORTING.csv", sep=",", low_memory=False
)
ids = pd.read_csv(str(Path.cwd()) + "\data\L_AIRPORT_ID.csv")
cs = pd.read_csv(str(Path.cwd()) + "\data\L_UNIQUE_CARRIERS.csv")

# prepare the data for the models
df, feature_columns_dep = bd.process_data(df)
df, feature_columns_arr = ba.process_data(df)

# if the models folder does not exist, create a folder called "models"
# if the models folder is empty, uncomment this:
# dep_model, _, _ = bd.train_model(df, feature_columns_dep)
# joblib.dump(dep_model, "./models/dep_model.joblib")
# arr_model, _, _ = bd.train_model(df, feature_columns_arr)
# joblib.dump(dep_model, "./models/arr_model.joblib")

# load in the models from the models folder
dep_model = joblib.load("./models/dep_model.joblib")
arr_model = joblib.load("./models/arr_model.joblib")

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# create the overview plots
calendar_fig = create_heatmap(df)
full_calendar_fig = create_heatmap(df, full=True)
feature_fig = create_feature_overview(df, "OP_UNIQUE_CARRIER", cs, ids)
time_fig = create_clock_heatmap(df)
causes_fig = create_causes_chart(df)

# create the feature importance models
dep_fig = create_feature_importance_graph(dep_model, df[feature_columns_dep])
arr_fig = create_feature_importance_graph(arr_model, df[feature_columns_arr])

# create the data for the dropdowns
origindata = helpers.create_origin_data(df, ids)
destdata = helpers.create_dest_data(df, ids)
carrierdata = helpers.create_carrier_data(df, cs)
intervals = helpers.create_interval_data()
intervals_arr = helpers.create_interval_arr_data()
features = helpers.create_feature_data()

# layout part 1: the top row
topbar = dbc.Row(
    [
        # title
        dbc.Row(
            [
                dbc.Col([], width=4),
                dbc.Col(
                    html.H1("FLIGHTSAVER", id="title"),
                    style={"textAlign": "center"},
                    width=4,
                ),
                # calendar buttons
                dbc.Col(
                    [
                        # full calendar button
                        dbc.Row([], style={"height": "20%"}),
                        dbc.Row(
                            html.Div(
                                [
                                    dbc.Button("Year calendar", id="open", n_clicks=0),
                                    dbc.Modal(
                                        [
                                            dbc.ModalHeader(dbc.ModalTitle("Calendar")),
                                            dbc.ModalBody(
                                                [
                                                    dcc.Graph(
                                                        id="cal-heatmap",
                                                        figure=full_calendar_fig,
                                                        config={
                                                            "displayModeBar": False
                                                        },
                                                    )
                                                ]
                                            ),
                                            dbc.ModalFooter(
                                                dbc.Button(
                                                    "Close", id="close", n_clicks=0
                                                )
                                            ),
                                        ],
                                        id="modal",
                                        is_open=False,
                                        size="lg",
                                    ),
                                ]
                            ),
                            style={"height": "77%"},
                        ),
                    ],
                    width=2,
                ),
                dbc.Col(
                    [
                        # remove selection button
                        dbc.Row([], style={"height": "20%"}),
                        dbc.Row(
                            dbc.Button(
                                "Remove selection",
                                id="remove-sel",
                                style={"display": "none"},
                            ),
                            style={"height": "77%"},
                        ),
                    ],
                    width=2,
                ),
            ],
            style={"height": "15%"},
        ),
        # input
        dbc.Row(
            [
                # origin and destination dropdowns
                dbc.Col(
                    [
                        dbc.Row([], style={"height": "10%"}),
                        dbc.Row(
                            html.Div(
                                "Select your flight details", className="text-standard"
                            ),
                            style={"textAlign": "center", "height": "10%"},
                        ),
                        dbc.Row([], style={"height": "5%"}),
                        dbc.Row(
                            dcc.Dropdown(
                                origindata,
                                id="start-dropdown",
                                placeholder="Select a starting point",
                                style={"height": "32%"},
                                optionHeight=100,
                            ),
                            style={"font-family": "Arial, Helvetica, sans-serif"},
                        ),
                        dbc.Row(
                            dcc.Dropdown(
                                destdata,
                                id="dest-dropdown",
                                placeholder="Select a destination",
                                style={"margin-top": 16, "height": "32%"},
                                optionHeight=100,
                            ),
                            style={"font-family": "Arial, Helvetica, sans-serif"},
                        ),
                    ],
                    width=3,
                    style={"height": "100%"},
                ),
                # carrier dropdowns
                dbc.Col(
                    [
                        dbc.Row([], style={"height": "10%"}),
                        dbc.Row(
                            html.Div(
                                "Select your first carrier", className="text-standard"
                            ),
                            style={"textAlign": "center", "height": "10%"},
                        ),
                        dbc.Row([], style={"height": "2%"}),
                        dbc.Row(
                            dcc.Dropdown(
                                carrierdata,
                                id="carr-1-dropdown",
                                placeholder="Select the first carrier",
                                style={"margin-top": 2, "height": "23%"},
                            ),
                            style={"font-family": "Arial, Helvetica, sans-serif"},
                        ),
                        dbc.Row([], style={"height": "2%"}),
                        dbc.Row(
                            html.Div(
                                "Select your second carrier", className="text-standard"
                            ),
                            style={"textAlign": "center", "height": "10%"},
                        ),
                        dbc.Row([], style={"height": "2%"}),
                        dbc.Row(
                            dcc.Dropdown(
                                carrierdata,
                                id="carr-2-dropdown",
                                placeholder="Select the second carrier",
                                style={"margin-top": 5, "height": "23%"},
                            ),
                            style={"font-family": "Arial, Helvetica, sans-serif"},
                        ),
                        dbc.Row([], style={"height": "5%"}),
                    ],
                    width=3,
                    style={"height": "100%"},
                ),
                # mini calendar heatmap
                dbc.Col(
                    [
                        dbc.Row(
                            dcc.Graph(
                                id="mini-heatmap",
                                figure=calendar_fig,
                                config={"displayModeBar": False},
                            ),
                            style={"height": "82%"},
                        ),
                        dbc.Row([], style={"height": "15%"}),
                    ],
                    width=6,
                ),
            ],
            style={"height": "75%"},
        ),
        dbc.Row([], style={"height": "5%"}),
    ],
    id="top-bar",
    style={"height": "30vh"},
)

# layout 2: column with departure delay model plot
departure = dbc.Col(
    [
        dbc.Row([], style={"height": "2%"}),
        # column title
        dbc.Row(
            html.Div("Predicted departure delay (mins)"),
            style={"height": "4%"},
            className="text-standard",
        ),
        dbc.Row([], style={"height": "2%"}),
        # interval dropdown
        dbc.Row(
            dcc.Dropdown(intervals, id="dep-dropdown", value=1),
            style={"height": "5%", "font-family": "Arial, Helvetica, sans-serif"},
        ),
        dbc.Row([], style={"height": "4%"}),
        # departure delay prediction
        dbc.Row(
            dcc.Graph(
                id="dep",
                figure=go.Figure(),
            ),
            style={"height": "44%"},
        ),
        # departure delay feature importance
        dbc.Row(
            [
                dbc.Col([], width=2),
                dbc.Col(
                    dcc.Graph(id="dep-why", figure=dep_fig),
                    width=8,
                ),
                dbc.Col([], width=2),
            ],
            style={"height": "26%"},
        ),
    ],
    id="dep-col",
    width=6,
    style={"height": "70vh", "display": "none"},
)

# layout 3: column with arrival delay model plot
arrival = dbc.Col(
    [
        dbc.Row([], style={"height": "2%"}),
        # column title
        dbc.Row(
            html.Div("Predicted arrival delay (mins)"),
            style={"height": "4%"},
            className="text-standard",
        ),
        dbc.Row([], style={"height": "2%"}),
        # interval dropdown
        dbc.Row(
            dcc.Dropdown(intervals_arr, id="arr-dropdown", value=2),
            style={"height": "5%", "font-family": "Arial, Helvetica, sans-serif"},
        ),
        dbc.Row([], style={"height": "4%"}),
        # arrival delay prediction
        dbc.Row(
            dcc.Graph(id="arr", figure=go.Figure()),
            style={"height": "44%"},
        ),
        # arrival delay feature importance
        dbc.Row(
            [
                dbc.Col([], width=2),
                dbc.Col(
                    dcc.Graph(id="arr-why", figure=arr_fig),
                    width=8,
                ),
                dbc.Col([], width=2),
            ],
            style={"height": "26%"},
        ),
    ],
    id="arr-col",
    width=6,
    style={"height": "70vh", "display": "none"},
)

# layout 4: right hand column with overview plots
summaries = dbc.Col(
    [
        dbc.Row([], style={"height": "2%"}),
        dbc.Row(
            html.Div("Select feature for delay overview", className="text-standard"),
            style={"height": "3%", "textAlign": "center"},
        ),
        # dropdown with features for feature overview
        dbc.Row(
            dcc.Dropdown(features, id="feature-dropdown", value="OP_UNIQUE_CARRIER"),
            style={"height": "5%", "font-family": "Arial, Helvetica, sans-serif"},
        ),
        # bar chart with feature overview
        dbc.Row(
            dcc.Graph(id="feature-things", figure=feature_fig), style={"height": "37%"}
        ),
        # heatmap with time overview
        html.Div(
            dbc.Row(dcc.Graph(id="time-heatmap", figure=time_fig)),
            style={"height": "27%"},
            className="center-graph",
        ),
        # bar chart with delay causes
        html.Div(
            dbc.Row(
                dcc.Graph(id="delay-causes", figure=causes_fig), style={"height": "20%"}
            ),
            className="center-graph",
        ),
    ],
    id="sumbar",
    width=3,
    style={"height": "100vh"},
)

# full layout
app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col([topbar, dbc.Row([departure, arrival])], width=9),
                summaries,
            ],
            style={"height": "100vh"},
        ),
        dcc.Store(id="selectedDate"),  # stores the selected date
        dcc.Store(id="locations"),     # stores the selected locations
        dcc.Store(id="carriers"),      # stores the selected carriers
    ],
    fluid=True,
    style={"height": "100vh", "overflow": "hidden"},
)


# callback 1: heatmap update (both heatmaps)
@callback(
    Output("mini-heatmap", "figure"), # update the mini calendar
    Output("cal-heatmap", "figure"),  # update the full calendar
    Output("remove-sel", "style"),    # visualise the remove button
    Output("selectedDate", "data"),   # store the selected date
    Input("mini-heatmap", "clickData"),
    Input("cal-heatmap", "clickData"),
    Input("remove-sel", "n_clicks"),
    prevent_initial_call=True,
)
def update_minimap(click_data, click_data_full, b1):
    """
    Updates both the mini calendar and full calendar
    when a date is selected or the selection is removed

    :param click_data: the clickdata of the mini heatmap
    :param click_data_full: the clickdata of the full heatmap
    :param b1: the number of clicks on the remove selection button
    :returns: the mini calendar figure, the full calendar figure,
              the style of the remove selection button,
              and the selected date
    """

    # see which input triggered the callback
    trigger = ctx.triggered_id

    # get the clickData if it exists
    if trigger == "remove-sel":
        date = None
    elif trigger == "mini-heatmap":
        if click_data is not None:
            date = click_data["points"][0]["customdata"]
    elif trigger == "cal-heatmap":
        if click_data_full is not None:
            date = click_data_full["points"][0]["customdata"]
    else:
        date = None

    # get the new state of the remove selection button
    style = {"display": "none"}

    # check if there was a point clicked
    if date is not None:
        # point was clicked
        style = {"display": "inline-flex"}

        # update the mini calendar
        fig1 = create_heatmap(df, False, date)
        # update the full calendar
        fig2 = create_heatmap(df, True, date)

    else:
        # update the mini calendar
        fig1 = create_heatmap(df, False)
        # update the full calendar
        fig2 = create_heatmap(df, True)

    return fig1, fig2, style, date


# callback 2: open or close the full calendar popup
@callback(
    Output("modal", "is_open"),  # update whether the popup is open
    Input("open", "n_clicks"),
    Input("close", "n_clicks"),
    State("modal", "is_open"),
    prevent_initial_call=True,
)
def update_popup(open, close, is_open):
    """
    Updates the visibility of the full calendar popup

    :param open: the clicks on the Full calendar button
    :param close: the clicks on the close calendar button
    :param is_open: the current state of the popup
    :returns: the new state of the popup
    """
    # get which button triggered the callback
    trigger = ctx.triggered_id

    # if the callback was triggered by a button, update the state
    if trigger == "open" or trigger == "close":
        return not is_open
    else:
        return is_open


# callback 3: put in location details
@callback(
    Output("locations", "data"),       # update the selected locations
    Output("start-dropdown", "value"), # update the value of the origin dropdown
    Output("dest-dropdown", "value"),  # update the value of the destination dropdown
    Input("start-dropdown", "value"),
    Input("dest-dropdown", "value"),
    Input("remove-sel", "n_clicks"),
    prevent_initial_call=True,
)
def update_locations(start, dest, rem):
    """
    Updates the locations selected in the dropdowns

    :param start: the value of the origin dropdown
    :param dest: the value of the destination dropdown
    :param rem: the clicks on the remove selection button
    :returns: the new values for the dropdowns and the selected locations
    """
    # get the trigger of the callback
    trigger = ctx.triggered_id

    # if the trigger is the remove selection button,
    # set everything to None
    if trigger == "remove-sel":
        return (None, None), None, None
    # else, return the selected values, and do not update
    # the dropdowns to avoid a callback loop
    return (start, dest), no_update, no_update


# callback 4: put in carrier details
@callback(
    Output("carriers", "data"),         # update the selected carriers
    Output("carr-1-dropdown", "value"), # update the value of the first carrier dropdown
    Output("carr-2-dropdown", "value"), # update the value of the second carrier dropdown
    Input("carr-1-dropdown", "value"),
    Input("carr-2-dropdown", "value"),
    Input("remove-sel", "n_clicks"),
    prevent_initial_call=True,
)
def update_locations(c1, c2, rem):
    """
    Updates the carriers selected in the dropdowns

    :param c1: the value of the first carrier dropdown
    :param c2: the value of the second carrier dropdown
    :param rem: the clicks on the remove selection button
    :returns: the new values for the dropdowns and the selected carriers
    """
    # get the trigger of the callback
    trigger = ctx.triggered_id

    # if the trigger is the remove selection button,
    # set everything to None
    if trigger == "remove-sel":
        return (None, None), None, None
    # else, return the selected values, and do not update
    # the dropdowns to avoid a callback loop
    return (c1, c2), no_update, no_update


# callback 5: update time heatmap
@callback(
    Output("time-heatmap", "figure"),  # update the figure of the time heatmap
    Input("selectedDate", "data"),
    prevent_initial_call=True,
)
def update_time_map(date):
    """
    Updates the figure for the time heatmap

    :param date: the selected date
    :returns: the new figure for the time heatmap
    """

    # if there is a date selected, filter the data displayed
    # in the time heatmap to that date
    if date is not None:
        data = df.loc[(df["DAY_OF_MONTH"] == date[0]) & (df["MONTH"] == date[1])]
        return create_clock_heatmap(data)
    # else, just use the full dataset
    else:
        return create_clock_heatmap(df)


# callback 6: update departure delay barchart
@callback(
    Output("dep", "figure"),    # update the departure delay figure
    Output("dep-col", "style"), # update the visibility of this layout column
    Input("selectedDate", "data"),
    Input("locations", "data"),
    Input("carriers", "data"),
    Input("dep-dropdown", "value"),
    prevent_initial_call=True,
)
def update_dep_bar(date, locs, carrs, interval):
    """
    Updates the bar chart that shows the predictions
    for the departure delay

    :param date: the selected date
    :param locs: the selected locations
    :param carrs: the selected carriers
    :param interval: the selected interval to display
    :returns: the figure displaying predicted departure delay,
              and whether the column is visible
    """

    # make sure there is an interval selected,
    # otherwise use the standard 1 day
    if interval is None:
        interval = 1

    # if everything is complete, do prediction
    if date is not None and locs is not None and carrs is not None:
        if (
            locs[0] is not None
            and locs[1] is not None
            and carrs[0] is not None
            and carrs[1] is not None
        ):
            fig = bd.create_bargraph(
                dep_model,
                date[0],
                date[1],
                carrs[0],
                carrs[1],
                locs[0],
                locs[1],
                interval
            )
            return fig, {"display": "inline"}

    # if the input is not complete, make the column invisible
    return go.Figure(), {"display": "none"}


# callback 7: update arrival delay barchart
@callback(
    Output("arr", "figure"),    # update the departure delay figure
    Output("arr-col", "style"), # update the visibility of this layout column
    Input("selectedDate", "data"),
    Input("locations", "data"),
    Input("carriers", "data"),
    Input("arr-dropdown", "value"),
    prevent_initial_call=True,
)
def update_arr_bar(date, locs, carrs, interval):
    """
    Updates the bar chart that shows the predictions
    for the arrival delay

    :param date: the selected date
    :param locs: the selected locations
    :param carrs: the selected carriers
    :param interval: the selected interval to display
    :returns: the figure displaying predicted arrival delay,
              and whether the column is visible
    """

    # make sure there is an interval selected,
    # otherwise use the standard 2 days
    if interval is None:
        interval = 2

    # if everything is complete, do prediction
    if date is not None and locs is not None and carrs is not None:
        if (
            locs[0] is not None
            and locs[1] is not None
            and carrs[0] is not None
            and carrs[1] is not None
        ):
            fig = ba.create_bargraph(
                arr_model,
                date[0],
                date[1],
                carrs[0],
                carrs[1],
                locs[0],
                locs[1],
                interval
            )
            return fig, {"display": "inline"}

    # if the input is not complete, make the column invisible
    return go.Figure(), {"display": "none"}


# callback 8: update delay overview per feature
@callback(
    Output("feature-things", "figure"),  # update the feature overview plot
    Input("selectedDate", "data"),
    Input("locations", "data"),
    Input("carriers", "data"),
    Input("feature-dropdown", "value"),
)
def update_feature_chart(date, locs, carrs, feature):
    """
    Updates the feature overview plot

    :param date: the selected date
    :param locs: the selected locations
    :param carrs: the selected carriers
    :param feature: the selected feature
    :returns: the feature overview figure
    """

    # if there is a date, filter the dataset to that date
    data = df
    if date is not None:
        data = data.loc[(data["DAY_OF_MONTH"] == date[0]) & (data["MONTH"] == date[1])]

    # if a feature is selected, display that feature
    if feature is not None:
        return create_feature_overview(data, feature, cs, ids, locs, carrs)
    # otherwise, display the default feature: carrier
    else:
        return create_feature_overview(data, "OP_UNIQUE_CARRIER", cs, ids, locs, carrs)


# callback 9: update the departure/arrival dropdowns depending on each other
@callback(
    Output("dep-dropdown", "value"),  # update the departure dropdown
    Output("arr-dropdown", "value"),  # update the arrival dropdown
    Input("dep-dropdown", "value"),
    Input("arr-dropdown", "value"),
    prevent_initial_call=True,
)
def update_dropdowns(dep, arr):
    """
    Updates the values in the departure and arrival dropdowns
    when one of them is changed

    :param dep: the value in the departure dropdown
    :param arr: the value in the arrival dropdown
    :returns: the new values for both dropdowns
    """

    # get which dropdown triggered the callback
    trigger = ctx.triggered_id

    # check if they're both selected
    if dep is not None and arr is not None:
        if dep == arr - 1:
            # they are the same now, don't update them
            return no_update, no_update
        elif trigger == "dep-dropdown":
            # don't update the departure dropdown
            # to avoid callback loops
            return no_update, dep + 1
        elif trigger == "arr-dropdown":
            # don't update the arrival dropdown
            # to avoid callback loops
            return arr - 1, no_update
    # if they aren't, don't update them
    return no_update, no_update


# callback 10: update the delay causes bar chart when a date is selected
@callback(
    Output("delay-causes", "figure"),  # update the delay causes figure
    Input("selectedDate", "data"),
)
def update_causes(date):
    """
    Updates the figure displaying the delay causes

    :param date: the selected date
    :returns: the delay causes figure
    """

    # if a date is selected, filter the dataset to that date
    if date is not None:
        return create_causes_chart(
            df.loc[(df["DAY_OF_MONTH"] == date[0]) & (df["MONTH"] == date[1])]
        )
    # otherwise, use the full dataset
    return create_causes_chart(df)


if __name__ == "__main__":
    app.run()
