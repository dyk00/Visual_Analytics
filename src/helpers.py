import pandas as pd


def create_origin_data(df, ids):
    """
    Creates the dropdown data for all origin airports in the data

    :param df: the dataset to use
    :param ids: the ids and names for all airports
    :returns: the dropdown data for all destinations
    """
    # get all unique origin airports in the data
    unique_origins = list(df["ORIGIN_AIRPORT_ID"].dropna().unique())

    # use the text from the airport data
    unique_text = [
        list(ids.loc[ids["Code"] == origin]["Description"])[0]
        for origin in unique_origins
    ]

    # return the text as label and the id as value for all origin airports
    return [{"label": text, "value": i} for i, text in zip(unique_origins, unique_text)]


def create_dest_data(df, ids):
    """
    Creates the dropdown data for all destination airports in the data

    :param df: the dataset to use
    :param ids: the ids and names for all airports
    :returns: the dropdown data for all destinations
    """
    # get all unique origin airports in the data
    unique_dests = list(df["DEST_AIRPORT_ID"].dropna().unique())

    # use the text from the airport data
    unique_text = [
        list(ids.loc[ids["Code"] == dest]["Description"])[0] for dest in unique_dests
    ]

    # return the text as label and the id as value for all destination airports
    return [{"label": text, "value": i} for i, text in zip(unique_dests, unique_text)]


def create_carrier_data(df, cs):
    """
    Creates the dropdown data for each carrier in the data,
    including its abbreviation

    :param df: the dataset to use
    :param cs: the ids and names for all carriers
    :returns: the dropdown data for the carriers
    """
    # get all unique carriers in the data
    unique_carriers = list(df["OP_UNIQUE_CARRIER"].dropna().unique())

    # use the text from the airport data and add the id
    unique_text = [
        list(cs.loc[cs["Code"] == carrier]["Description"])[0] + " (" + carrier + ")"
        for carrier in unique_carriers
    ]

    # return the text and id as label and the id as value for all carriers
    return [
        {"label": text, "value": i} for i, text in zip(unique_carriers, unique_text)
    ]


def create_interval_data():
    """
    Creates the data for the intervals in the
    departure delay dropdown

    :returns: the dropdown data with 1 and 3 days
    """
    return [{"label": "1 day", "value": 1}, {"label": "3 days", "value": 3}]


def create_interval_arr_data():
    """
    Creates the data for the intervals in the
    arrival delay dropdown

    :returns: the dropdown data with 2 and 4 days
    """
    return [{"label": "2 days", "value": 2}, {"label": "4 days", "value": 4}]


def create_feature_data():
    """
    Creates the data for the features to display in the
    feature overview dropdown

    :returns: the dropdown data with origin, destination and carrier
    """
    return [
        {"label": "Carrier", "value": "OP_UNIQUE_CARRIER"},
        {"label": "Starting point", "value": "ORIGIN_AIRPORT_ID"},
        {"label": "Destination", "value": "DEST_AIRPORT_ID"},
    ]
