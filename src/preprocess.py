import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.stats import pearsonr

def dataset_info(df):
    """
    Print the basic information and details of a given DataFrame.

    Parameters
    ----------
    ds : pandas.DataFrame
        The DataFrame whose information is to be printed.
    
    Description
    -----------

    """
    # Exception handling for non-DataFrame inputs
    try:
        # Print Dataset Info
        print('\033[36m\033[1m' + '.: Dataset Info :.' + '\033[0m')
        print('\033[0m\033[36m*' * 20)
        print('\033[0m' + 'Total Rows:' + '\033[36m\033[1m', df.shape[0])
        print('\033[0m' + 'Total Columns:' + '\033[36m\033[1m', df.shape[1])
        print('\033[0m\033[36m*' * 20)
        print('\n')

        # Print Dataset Detail
        print('\033[1m' + '.: Dataset Details :.')
        print('\033[0m\033[36m*' * 22 + '\033[0m')
        df.info(memory_usage=False)

    except:
        # Print error message
        print("Please input DataFrame objects.")

def skewness_checking(df, type):
    """
    Conduct a skewness check for continuous columns in a DataFrame. This function calculates and prints the skewness of each numerical column, helping to identify the asymmetry in the data distribution.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe which contains the data for skewness analysis.
    type : str
        A descriptive string that is used in the printed output to specify the type of data being analyzed. For example, this could be 'Credit', 'Debit', etc., to indicate the type of financial data being analyzed.
    
    Description
    -----------
    The function prints the skewness values of the DataFrame's numerical columns. Skewness close to zero indicates a symmetric distribution, while a significant deviation from zero suggests an asymmetry. 
    Positive skewness indicates a distribution with an asymmetric tail extending towards more positive values, and negative skewness indicates a distribution with an asymmetric tail extending towards more negative values. 
    This analysis is crucial in data preprocessing to decide on potential transformations for normalizing data.

    """
    # Print a formatted header to indicate the type of skewness being checked
    print('\033[36m\033[1m'+' .: '+type+' Credit Continuous Columns Skewness :.'+'\033[0m')

    # Add exception handling if there are no numerical columns in the DataFrame
    try:
        # Calculate and print the skewness of each numerical column in the DataFrame
        print(df.select_dtypes(exclude='object').skew(axis = 0, skipna = True))
    except:
        # Print a message indicating that there are no numerical columns in the DataFrame
        print('No numerical columns in the DataFrame')

def outlier_handling(df, columns):
    """
    Handle outliers in specified columns of a DataFrame by capping them using the Interquartile Range (IQR) method.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe containing the data.
    columns : list
        The list of column names in the dataframe where outliers need to be handled.
    
    Description
    -----------
    This function iterates through each specified column and calculates the first (Q1) and third (Q3) quartiles, and the Interquartile Range (IQR). 
    It then identifies outliers as those values that fall below Q1 - 1.5 * IQR or above Q3 + 1.5 * IQR. These outliers are then capped to the lower and upper whiskers respectively, mitigating their impact on the dataset.
    
    """
    # Iterate over each specified column to handle outliers
    for column in columns:
        # Calculate the first quartile (Q1)
        Q1 = df[column].quantile(0.25)
        # Calculate the third quartile (Q3)
        Q3 = df[column].quantile(0.75)
        # Calculate the Interquartile Range (IQR)
        IQR = Q3 - Q1

        # Define lower and upper whiskers for outlier detection
        Lower_whisker = Q1 - 1.5 * IQR
        Upper_whisker = Q3 + 1.5 * IQR

        # Cap values below the lower whisker to the lower whisker value
        df.loc[df[column] < Lower_whisker, column] = Lower_whisker
        # Cap values above the upper whisker to the upper whisker value
        df.loc[df[column] > Upper_whisker, column] = Upper_whisker

def interpolate_quart_to_month(df, column):
    """
    Interpolate data from a quarterly frequency to a monthly frequency using cubic spline interpolation.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe containing the quarterly data.
    column : str
        The name of the column in the dataframe that needs to be interpolated.

    Returns
    -------
    pandas.DataFrame
        A new dataframe with the interpolated monthly data. This dataframe contains two columns: 'date' for the monthly dates and the interpolated values of the specified column.

    Description
    -----------
    This function takes a dataframe with a time series column in a quarterly frequency and interpolates it to a monthly frequency using cubic spline interpolation. 
    The function generates a range of monthly dates based on the original data's date range and then applies cubic spline interpolation to estimate the values for these new monthly dates.

    """
    # Exception handling if the dataframe has no DateTime index
    try:
        # Create a cubic spline interpolator using the Julian dates and the specified column values
        quarterly_interpolator = CubicSpline(df.index.to_julian_date(), df[column])

        # Generate a range of monthly dates from the minimum to the maximum date in the original dataset
        monthly_dates = pd.date_range(start=df.index.min(), end=df.index.max(), freq='MS')

        # Apply the cubic spline interpolator to the generated monthly dates to get interpolated values
        monthly_data = quarterly_interpolator(monthly_dates.to_julian_date())

        # Create a new DataFrame with the monthly dates and the interpolated values
        monthly_household = pd.DataFrame({'date': monthly_dates, 'household_consumption': monthly_data})

        # Return the new DataFrame with interpolated values
        return monthly_household
    
    except:
        # Print an error message if the dataframe has no DateTime index
        print('The dataframe has no DateTime index')

def pearson_corr_features_name(df, target, threshold):
    """
    Identify and return the names of features in a dataframe that have a Pearson correlation coefficient 
    with a target variable greater than a specified threshold.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe containing the features and target variable.
    target : str
        The name of the target variable in the dataframe.
    threshold : float
        The threshold for the Pearson correlation coefficient. Features with a correlation coefficient 
        greater than this value with the target variable will be included.

    Returns
    -------
    list
        A list of feature names that have a Pearson correlation coefficient greater than the specified threshold 
        with the target variable.

    Description
    -----------
    This function computes the Pearson correlation coefficient between the target variable and other features in the dataframe. 
    It then filters and returns the names of those features whose correlation with the target variable is greater than the 
    specified threshold, indicating a significant linear relationship.

    """
    # Calculate the Pearson correlation coefficients of all features with the target variable
    testResult = pd.DataFrame(df.corr(method='pearson')[target].sort_values(ascending=False))

    # Initialize an empty list to store the names of features that meet the correlation threshold criteria
    corr_features = []

    # Iterate over the correlation results
    for i in range(len(testResult)):
        # Check if the correlation coefficient is greater than the specified threshold
        if testResult[target][i] > threshold:
            # Append the feature name to the list
            corr_features.append(testResult.index[i])

    # Remove the target variable from the list of correlated features
    corr_features.remove(target)

    # Return the list of correlated feature names
    return corr_features

def pearson_corr_rows(df, target, features, included_col=[]):
    """
    Calculate the Pearson correlation coefficients between a target variable and a list of feature columns, 
    and return the results in a long-format DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe containing the data.
    target : str
        The name of the target variable in the dataframe.
    features : list
        The list of feature column names whose correlation with the target variable is to be calculated.
    included_col : list, optional
        Additional columns to be included in the final result DataFrame.

    Returns
    -------
    pandas.DataFrame
        A long-format DataFrame containing the correlation coefficients of each feature with the target variable.

    Description
    -----------
    This function computes the Pearson correlation coefficient between the target variable and a list of specified features. 
    The results are stored in a long-format DataFrame, which is useful for further analysis or visualization of the correlations.

    """
    # Create DataFrame to store the results
    correlation_df = df[included_col].copy()

    # Iterate through each feature column
    for feature_column in features:
        
        # Calculate Pearson correlation coefficient and p-value
        correlation_coefficient, p_value = pearsonr(df[target], df[feature_column])

        # Store the correlation coefficient in the result DataFrame
        correlation_df[feature_column] = round(correlation_coefficient,2)

    # Melt the DataFrame to convert it from wide to long format
    melted_correlation_df = pd.melt(correlation_df, id_vars=included_col, var_name='covariates', value_name='correlation_coefficient')

    # Drop duplicate rows
    melted_correlation_df.drop_duplicates()

    # Return the result DataFrame
    return melted_correlation_df
