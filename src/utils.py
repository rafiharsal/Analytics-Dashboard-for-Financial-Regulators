import pandas as pd
import matplotlib.pyplot as plt
import urllib
from sqlalchemy import create_engine

def pearson_corr_test(df, target):
    """
    Calculate the Pearson correlation coefficients for a target variable with other variables in a DataFrame.

    Parameters:
    df : DataFrame
        The DataFrame containing the dataset.
    target : str
        The name of the target variable for which the correlation is to be calculated.

    Returns:
    DataFrame
        A DataFrame with correlation values of the target variable with other variables, sorted in descending order.

    """
    testResult = pd.DataFrame(df.corr(method='pearson')[target].sort_values(ascending=False))
    return testResult

def split_len(ts, percent):
    """
    Determine the index for splitting a TimeSeries based on a specified percentage.

    Parameters:
    ts : TimeSeries
        The TimeSeries data to be split.
    percent : float
        The percentage at which the TimeSeries should be split.

    Returns:
    int
        The index where the TimeSeries should be split.

    """
    split_ratio = percent
    split_index = round(int(len(ts) * split_ratio))
    return split_index

def month_diff_ts(ts1, ts2):
    """
    Calculate the difference in months between the last dates of two TimeSeries.

    Parameters:
    ts1, ts2 : TimeSeries
        The TimeSeries to compare.

    Returns:
    int
        The difference in months between the last month of ts1 and ts2.

    """
    # Exception handling for non-TimeSeries inputs
    try:
        # Get the last month in each TimeSeries
        ts1_last_month = ts1.time_index[-1]
        ts2_last_month = ts2.time_index[-1]

        # Calculate the difference in months between the last month in each TimeSeries
        diff = (ts2_last_month.year - ts1_last_month.year) * 12 + (ts2_last_month.month - ts1_last_month.month)

        # Return the difference in months
        return diff
    
    except:
        # Print error message
        print("Please input TimeSeries objects.")

def load_dw(server, dw_name):
    """
    Connect to a data warehouse using SQL Server.

    Parameters:
    server : str
        The server address.
    dw_name : str
        The name of the data warehouse.

    Returns:
    Engine
        SQLAlchemy engine representing the connection to the data warehouse.

    """
    # Exception handling for invalid server address
    try:
        # Specify the connection string to connect to the database
        connection = (
                    r'DRIVER={ODBC Driver 17 for SQL Server};'
                    r'SERVER='+server+';'
                    r'DATABASE='+dw_name+';'
                    r'Trusted_Connection=yes;')
        
        # Parse the connection string    
        quoted = urllib.parse.quote_plus(connection)

        # Create the connection
        engine = create_engine('mssql+pyodbc:///?odbc_connect={}'.format(quoted))

        # Print success message
        print("Successfully connected to the data warehouse.")

        return engine
    
    except:
        # Print error message
        print("Invalid server address.")