# plotly
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from plotly.express.colors import sample_colorscale, make_colorscale

# other libraries
import pandas as pd
import numpy as np
import datetime

# to turn the month index into the month name
months = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}


def create_heatmap(df, full=False, selectedcell=None):
    """
    Creates a mini calendar heatmap of 4 months
    that allows selection of a date

    :param df: the dataset to display
    :param selectedcell: the selected date in the heatmap
    :returns: a plotly heatmap figure in the shape of a
              mini calendar
    """

    # get current month to show the 4 months this one is part of
    if selectedcell is not None:
        this_month = selectedcell[1]
    else:
        this_month = datetime.datetime.now().month

    # get the months to display
    if not full:
        # mini calendar
        if this_month <= 4:
            mnth = 1
        elif this_month >= 9:
            mnth = 9
        else:
            mnth = 5
        endmonth = mnth + 3
        rows = 1
    else:
        # full calendar
        mnth = 1
        endmonth = 12
        rows = 3

    # get the current year

    # since we have data for 2023, we display 2023
    year = 2023
    # if you want the current year displayed, uncomment the following:
    # year = datetime.datetime.now().year

    # create the figure with 4 subplots per row
    fig = make_subplots(
        rows=rows,
        cols=4,
        subplot_titles=[months[i] for i in range(mnth, endmonth + 1)],
        shared_xaxes=False,
    )

    # add the traces for each month to the figure
    col = 1
    row = 1
    for m in range(mnth, endmonth + 1):
        # is this the first month, flag it for that
        if col == 1 and row == 1:
            first = True
        else:
            first = False

        # create the heatmap for every month
        fig1, colour = create_subheatmap_month(df, m, year, selectedcell, first=first)
        fig.add_trace(fig1, row=row, col=col)

        # if the selected date is in this month, add a heatmap
        # to this month as overlay to highlight it
        if selectedcell is not None and selectedcell[1] == m:
            fig2, _ = create_subheatmap_month(df, m, year, selectedcell, True, colour)
            fig.add_trace(fig2, row=row, col=col)

        # go to next month's subplot
        col += 1
        if col > 4:
            col = 1
            row += 1

    # create the layout of the figures
    if full:
        # full calendar (slightly larger)
        fig.update_layout(
            width=600,
            height=600,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_family="Arial, Helvetica, sans-serif",
            title="Average delay per day (mins). Select your flight date.",
        )
        fig.update_annotations(yshift=30)
    else:
        # mini calendar (smaller)
        fig.update_layout(
            width=500,
            height=250,
            margin={"l": 5, "r": 5},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_family="Arial, Helvetica, sans-serif",
            title="Average delay per day (mins). Select your flight date.",
            title_x=0.5,
            title_y=0.87,
        )
        fig.update_annotations(yshift=35)

    # update the x-axis of the heatmap to show the days of the week
    fig.update_xaxes(
        showline=False,
        showgrid=False,
        zeroline=False,
        tickmode="array",
        ticktext=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        tickvals=[1, 2, 3, 4, 5, 6, 7],
        side="top",
        fixedrange=True,
    )

    # update the y-axis of the heatmap not to show anything
    fig.update_yaxes(
        showline=False,
        showgrid=False,
        zeroline=False,
        autorange="reversed",
        showticklabels=False,
        fixedrange=True,
    )

    return fig


def create_subheatmap_month(
    df, month, year, selectedcell, selectmap=False, colour=None, first=False
):
    """
    Creates a heatmap for a single month, choosing a color map depending
    on whether a date is selected or not

    :param df: the dataset to use
    :param month: the month for which to generate the heatmap
    :param year: the year for which to generate the heatmap
    :param selectedcell: the date currently selected
    :param selectmap: if this is the overlay heatmap for the selected date
    :param colour: the colour used for the selected date
    :param first: whether this is the first map in the heatmap
    :returns: a plotly heatmap for this month
    """

    # get the dates for this month
    endday = 31
    if month == 2:
        endday = 28
    elif month == 4 or month == 6 or month == 9 or month == 11:
        endday = 30

    # use the start and end date of the datetime library
    start = datetime.date(year, month, 1)
    end = datetime.date(year, month, endday)

    # create the list of dates for this month
    diff = end - start
    dates = []
    cellpoint = None
    for i in range(diff.days + 1):
        dates.append(start + datetime.timedelta(i))
        if (
            selectedcell is not None
            and selectedcell[1] == month
            and dates[-1].day == selectedcell[0]
        ):
            # this is the position of the currently selected date
            cellpoint = i

    # get the values on the axes of the heatmaps
    weekofyear = [int(date.strftime("%W")) for date in dates]
    dayofweek = [date.weekday() + 1 for date in dates]

    # get the average delay for the heatmap colour
    if selectmap:
        # if this is the heatmap that is the overlay for a selected date,
        # we only want that specific date to have a value
        totaldelay = []
        for i in range(len(dates)):
            date = dates[i]
            if i == cellpoint:
                totaldelay.append(
                    df.loc[
                        (df["MONTH"] == int(date.strftime("%m")))
                        & (df["DAY_OF_MONTH"] == int(date.strftime("%d")))
                    ]["DEP_DELAY_NEW"].mean()
                )
            else:
                totaldelay.append(np.NaN)
    else:
        # otherwise, everything should have its actual average delay
        totaldelay = [
            df.loc[
                (df["MONTH"] == int(date.strftime("%m")))
                & (df["DAY_OF_MONTH"] == int(date.strftime("%d")))
            ]["DEP_DELAY_NEW"].mean()
            for date in dates
        ]

    # colormaps: 1 is before selection (and the selected date),
    # 2 is after selection for the dates that are not selected
    colmap1 = make_colorscale(["rgb(0,102,255)", "rgb(255,204,0)"])
    colmap2 = make_colorscale(["rgb(179,209,255)", "rgb(255,240,179)"])

    # there is a selected cell, but it this is not the overlay
    # in that case, we use the light colormap
    if (selectedcell is not None) and (not selectmap):
        # position in heatmap percentage for selected item
        if selectedcell[1] == month:
            pos = (totaldelay[cellpoint] - min(totaldelay)) / max(totaldelay)
            colour = sample_colorscale(colmap1, [pos])
        colourmap = colmap2

    # this is the overlay, so we use its specific color
    elif selectmap:
        colourmap = [(0, colour[0]), (1, colour[0])]

    # there is no date selected, so we use the bright colormap
    else:
        colourmap = colmap1

    # get the hovertext for the figure
    hovertext = [
        str(date.strftime("%B"))
        + " "
        + str(int(date.strftime("%d")))
        + ": "
        + str(round(delay, 1))
        + " minutes"
        for date, delay in zip(dates, totaldelay)
    ]

    # get the custom data so we can access it when we click on something
    customdata = [
        (int(date.strftime("%d")), int(date.strftime("%m"))) for date in dates
    ]

    # value for showscale
    if selectmap:
        sel = False
        colorticks = None
    elif (selectedcell is not None) and (not selectmap) and first:
        sel = True
        colorticks = {"nticks": 5}
    elif first:
        sel = True
        colorticks = {"nticks": 5}
    else:
        sel = False
        colorticks = None

    # generate heatmap and return
    return (
        go.Heatmap(
            y=weekofyear,
            x=dayofweek,
            z=totaldelay,
            text=hovertext,
            hoverinfo="text",
            xgap=3,
            ygap=3,
            showscale=sel,
            colorbar=colorticks,
            colorscale=colourmap,
            customdata=customdata,
        ),
        colour,
    )
