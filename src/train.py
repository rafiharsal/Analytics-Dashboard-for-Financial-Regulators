# Import Libraries
import pickle
import numpy as np
import pandas as pd
import math
from darts.metrics import mape, rmse, mae
from darts.dataprocessing.transformers import Scaler
scaler = Scaler()

def eval_model(model, len, train, val):
    """
    Evaluate a machine learning model on time series data.

    Parameters
    ----------
    model : TimeSeries model
        The time series model to be evaluated.
    len : int
        The length of the forecast horizon for predictions.
    train : TimeSeries
        The training dataset used to fit the model.
    val : TimeSeries
        The validation dataset used to evaluate the model.

    Description
    -----------
    This function evaluates a time series model's performance by fitting it on the training data, 
    making predictions, and then calculating and normalizing key performance metrics. It prints out 
    the Mean Absolute Percentage Error (MAPE), Root Mean Square Error (RMSE), Mean Absolute Error (MAE), 
    and a composite model score for an easy understanding of the model's performance.

    """
    # Exception handling for input with non TimeSeries data
    try:
        # Fit the model with the training data
        model.fit(train)

        # Predict future values based on the specified forecast horizon
        pred = model.predict(len)

        # Calculate the MAPE, RMSE, and MAE for the model predictions against the validation data
        mape_val = mape(pred, val)
        rmse_val = rmse(pred, val) / 1e9  # Normalize RMSE to a smaller scale
        mae_val = mae(pred, val) / 1e9   # Normalize MAE to a smaller scale

        # Calculate dividers for normalizing RMSE and MAE values
        rmse_divider = 10 ** round(math.log10(rmse_val))
        mae_divider = 10 ** round(math.log10(mae_val))

        # Normalize MAPE, RMSE, and MAE scores for a balanced comparison
        mape_score = mape_val / 100  # Convert MAPE from percentage to decimal
        rmse_score = rmse_val / rmse_divider  # Normalize RMSE score
        mae_score = mae_val / mae_divider    # Normalize MAE score

        # Calculate an overall model score based on the normalized metrics
        model_score = round(4 * mape_score + 3 * rmse_score + 3 * mae_score, 1)

        # Print the evaluation metrics and the composite model score
        print(f'{model} - MAPE: {mape_val:.2f}%, RMSE: {rmse_val:.2f}M, MAE: {mae_val:.2f}M, Score: {model_score:.1f}')

    except:
        print('Please input TimeSeries data.')

def eval_model_ml(model, len, train, val, cov):
    """
    Evaluate a machine learning model on time series data with covariates.

    Parameters
    ----------
    model : Machine Learning model
        The machine learning model to be evaluated.
    len : int
        The length of the forecast horizon for predictions.
    train : TimeSeries
        The training dataset used to fit the model.
    val : TimeSeries
        The validation dataset used to evaluate the model.
    cov : TimeSeries
        The covariates dataset to be used alongside the training data.

    Description
    -----------
    This function evaluates a machine learning model's performance by fitting it on 
    the training data (with covariates), making predictions, and then calculating 
    and normalizing key performance metrics. It prints out the Mean Absolute 
    Percentage Error (MAPE), Root Mean Square Error (RMSE), Mean Absolute Error (MAE), 
    and a composite model score for an easy understanding of the model's performance.

    """
    # Exception handling for input with non TimeSeries data
    try:
        # Scale the training data and covariates
        scaled_train, scaled_cov = scaler.fit_transform([train, cov])

        # Fit the model with the scaled training data and covariates
        model.fit(scaled_train, future_covariates = scaled_cov)

        # Predict future values using the model with covariates
        pred_scaled = model.predict(len, future_covariates = scaled_cov)
        
        # Inverse transform the predictions to their original scale
        pred = scaler.inverse_transform(pred_scaled)

        # Calculate the MAPE, RMSE, and MAE for the model predictions against the validation data
        mape_val = mape(pred, val)
        rmse_val = rmse(pred, val) / 1e9  # Normalize RMSE to a smaller scale
        mae_val = mae(pred, val) / 1e9   # Normalize MAE to a smaller scale

        # Calculate dividers for normalizing RMSE and MAE values
        rmse_divider = 10 ** round(math.log10(rmse_val))
        mae_divider = 10 ** round(math.log10(mae_val))

        # Normalize MAPE, RMSE, and MAE scores for a balanced comparison
        mape_score = mape_val / 100  # Convert MAPE from percentage to decimal
        rmse_score = rmse_val / rmse_divider  # Normalize RMSE score
        mae_score = mae_val / mae_divider    # Normalize MAE score

        # Calculate an overall model score based on the metrics
        model_score = round(4 * mape_score + 3 * rmse_score + 3 * mae_score, 1)

        # Print the evaluation metrics and the composite model score
        print(f'{model} - MAPE: {mape_val:.2f}%, RMSE: {rmse_val:.0f}M, MAE: {mae_val:.0f}M, Score: {model_score:.1f}')

    except:
        print('Please input TimeSeries data.')

def model_pkl(model, pickle_name, filepath=r"C:\Users\ASUS\OneDrive - Universiti Malaya\Sem 7\FYP\Model pkl\\"):
    """
    Save a machine learning model to a file using pickle.

    Parameters
    ----------
    model : Machine Learning model
        The machine learning model to be saved.
    pickle_name : str
        The name to be used for the saved model file.
    filepath : str, optional
        The directory path where the model file will be saved. Default is set to a specific path.

    Description
    -----------
    This function serializes a given machine learning model using pickle and saves it to a specified 
    directory with a given file name. This is useful for saving trained models for later use or deployment 
    without the need to retrain them.

    """
    # Exception handling for invalid filepaths
    try:
        # Construct the full filename with path and .pkl extension
        filename = pickle_name + '.pkl'

        # Serialize the model and save it to the specified filefilename = pickle_name + '.pkl'
        pickle.dump(model, open(filepath+filename, 'wb'))

        # Print a confirmation message
        print(f'{pickle_name} saved to {filepath}')

    except:
        # Print an error message if the filepath is invalid
        print('Please enter valid filepath.')  

def pred_model(model, len, train):
    """
    Fit a model to training data and predict future values.

    Parameters
    ----------
    model : TimeSeries model
        The time series model to be used for predictions.
    len : int
        The length of the forecast horizon.
    train : TimeSeries
        The training dataset used to fit the model.

    Description
    -----------
    This function trains a provided time series model on a given dataset and then uses the model to make 
    predictions for a specified number of future periods. The function returns the predicted values.

    Returns
    -------
    prediction : TimeSeries
        The predicted values for the specified number of future periods.

    """
    # Exception handling for input with non TimeSeries data
    try:
        # Fit the model to the training data
        model.fit(train)

        # Use the fitted model to predict future values for the specified length
        prediction = model.predict(len)

        # Return the prediction series
        return prediction
    
    except:
        # Print an error message if the input is not TimeSeries data
        print('Please input TimeSeries data.')

def pred_ml_model(model, len, train, cov):
    """
    Fit a machine learning model to scaled training data with covariates and make predictions.

    Parameters
    ----------
    model : Machine Learning model
        The machine learning model to be used for predictions.
    len : int
        The length of the forecast horizon.
    train : TimeSeries
        The training dataset used to fit the model.
    cov : TimeSeries
        The covariates dataset to be used alongside the training data.

    Description
    -----------
    This function first scales the training data and covariates using a predefined scaler. It then 
    fits the provided machine learning model to the scaled training data along with the scaled covariates. 
    After fitting, it uses the model to make predictions for a specified number of future periods. 
    The predictions are then inverse transformed to bring them back to their original scale. 
    The function returns these predictions.

    Returns
    -------
    prediction : TimeSeries
        The predicted values for the specified number of future periods, inverse scaled to the original scale.

    """
    # Exception handling for input with non TimeSeries data
    try:
        # Scale the training data and covariates
        scaled_train, scaled_cov = scaler.fit_transform([train, cov])

        # Fit the model with the scaled training data and covariates
        model.fit(scaled_train, future_covariates=scaled_cov)

        # Predict future values using the fitted model and scaled covariates
        prediction_scaled = model.predict(len, future_covariates=scaled_cov)

        # Inverse transform the predictions to bring them back to their original scale
        prediction = scaler.inverse_transform(prediction_scaled)

        # Return the inverse transformed prediction series
        return prediction
    
    except:
        # Print an error message if the input is not TimeSeries data
        print('Please input TimeSeries data.')

def out_pred_df(model, len, train, type_name, model_name, col_id='', start_val=0.8):
    """
    Generate a DataFrame with model predictions, backtesting results, and performance metrics.

    Parameters
    ----------
    model : TimeSeries model
        The time series model to be used for predictions and backtesting.
    len : int
        The length of the forecast horizon.
    train : TimeSeries
        The training dataset used for model fitting.
    type_name : str
        A descriptive name for the type of data or prediction.
    model_name : str
        The name of the model used for identification.
    col_id : str, optional
        Additional identifier to append to column names.
    start_val : float, optional
        The start point for backtesting as a fraction of the training data.

    Description
    -----------
    This function fits a time series model to training data, performs predictions, conducts backtesting, 
    and calculates evaluation metrics. It returns a DataFrame that includes both actual and predicted values, 
    year-over-year and month-over-month growth calculations, and model performance metrics.

    Returns
    -------
    combined_df : DataFrame
        A DataFrame that combines actual data, model predictions, backtesting results, and metrics.

    """
    # Fit the model with the training data
    model.fit(train)

    # Predict future values based on the specified forecast horizon
    pred_ts = model.predict(len)

    # Conduct backtesting on the model
    pred_backtest = model.historical_forecasts(train, start=start_val, forecast_horizon=1, verbose=True)

    # Convert the predicted and backtested TimeSeries to DataFrames
    pred_future_df = pred_ts.pd_dataframe()
    pred_backtest_df = pred_backtest.pd_dataframe()

    # Merge the predicted and backtested DataFrames
    pred_df = pd.concat([pred_backtest_df, pred_future_df])
   
    # Add 'type' and 'model' columns to pred_df for identification
    pred_df['type'] = type_name
    pred_df['model'] = model_name

    # Calculate evaluation metrics for the backtested predictions
    mape_val = mape(pred_backtest, train) / 100  # Convert percentage to decimal
    rmse_val = rmse(pred_backtest, train)
    mae_val = mae(pred_backtest, train)

    # Normalize RMSE and MAE scores
    rmse_divider = 10 ** round(math.log10(rmse_val))
    mae_divider = 10 ** round(math.log10(mae_val))
    rmse_score = rmse_val / rmse_divider
    mae_score = mae_val / mae_divider

    # Calculate an overall model score
    model_score = round(4 * (mape_val / 100) + 3 * rmse_score + 3 * mae_score, 1)

    # Add metric scores to pred_df
    pred_df['mape' + col_id] = round(mape_val, 2)
    pred_df['rmse' + col_id] = round(rmse_val)
    pred_df['mae' + col_id] = round(mae_val)
    pred_df['model_score' + col_id] = model_score

    # Calculate year-over-year and month-over-month growth for pred_df
    value_col_name = pred_df.columns[0]
    pred_df['yoy_growth' + col_id] = round(pred_df[value_col_name].pct_change(12) * 100, 2) / 100
    pred_df['mom_growth' + col_id] = round(pred_df[value_col_name].pct_change(1) * 100, 2) / 100

    # Convert actual training data to DataFrame
    actual_df = train.pd_dataframe()

    # Add columns for identification and metrics to actual_df
    actual_df['type'] = type_name
    actual_df['model'] = 'Actual'
    actual_df['mape' + col_id] = np.nan
    actual_df['rmse' + col_id] = np.nan
    actual_df['mae' + col_id] = np.nan
    actual_df['model_score' + col_id] = np.nan

    # Calculate year-over-year and month-over-month growth for actual_df
    actual_df['yoy_growth' + col_id] = round(actual_df[value_col_name].pct_change(12) * 100, 2) / 100
    actual_df['mom_growth' + col_id] = round(actual_df[value_col_name].pct_change(1) * 100, 2) / 100

    # Combine the actual and predicted data into a single DataFrame
    combined_df = pd.concat([actual_df, pred_df])

    # Reorder columns in combined_df and reset index with 'date' as the index name
    combined_df = combined_df[[combined_df.columns[1], combined_df.columns[2], combined_df.columns[0],
                               combined_df.columns[3], combined_df.columns[4], combined_df.columns[5],
                               combined_df.columns[6], combined_df.columns[7], combined_df.columns[8]]]
    combined_df = combined_df.reset_index().rename(columns={'index': 'date'})

    # Return the combined DataFrame
    return combined_df

def out_pred_ml_df(model, len, train, cov, type_name, model_name, start_val=0.8):
    """
    Generate a DataFrame with machine learning model predictions, backtesting results, and performance metrics.

    Parameters
    ----------
    model : Machine Learning model
        The machine learning model to be used for predictions and backtesting.
    len : int
        The length of the forecast horizon.
    train : TimeSeries
        The training dataset used for model fitting.
    cov : TimeSeries
        The covariates dataset used along with the training data.
    type_name : str
        A descriptive name for the type of data or prediction.
    model_name : str
        The name of the model used for identification.
    start_val : float, optional
        The start point for backtesting as a fraction of the training data.

    Description
    -----------
    This function fits a machine learning model to scaled training data with covariates, performs predictions, 
    conducts backtesting, and calculates evaluation metrics. It returns a DataFrame that includes both actual 
    and predicted values, year-over-year and month-over-month growth calculations, and model performance metrics.

    Returns
    -------
    combined_df : DataFrame
        A DataFrame that combines actual data, model predictions, backtesting results, and metrics.

    """
    # Scale training data and covariates before fitting the model
    scaled_train, scaled_cov = scaler.fit_transform([train, cov])

    # Fit the model with the scaled training data and covariates
    model.fit(scaled_train, future_covariates = scaled_cov)

    # Predict future values using the fitted model and covariates
    pred_ts_scaled = model.predict(len, future_covariates = scaled_cov)

    # Conduct backtesting on the scaled training data
    pred_backtest_scaled = model.historical_forecasts(scaled_train, future_covariates=cov, start=start_val, forecast_horizon=1, verbose=True)
    
    # Inverse transform the predictions to their original scale
    pred_ts = scaler.inverse_transform(pred_ts_scaled)
    pred_backtest = scaler.inverse_transform(pred_backtest_scaled)

    # Convert the predictions and backtesting results to DataFrames
    pred_future_df = pred_ts.pd_dataframe()
    pred_backtest_df = pred_backtest.pd_dataframe()

    # Merge the predicted and backtested DataFrames
    pred_df = pd.concat([pred_backtest_df, pred_future_df])

    # Add identification columns to the prediction DataFrame
    pred_df['type'] = type_name
    pred_df['model'] = model_name

    # Calculate and normalize evaluation metrics
    mape_val = mape(pred_backtest, train) / 100  # Convert percentage to decimal
    rmse_val = rmse(pred_backtest, train)
    mae_val = mae(pred_backtest, train)
    rmse_divider = 10 ** round(math.log10(rmse_val))
    mae_divider = 10 ** round(math.log10(mae_val))
    rmse_score = rmse_val / rmse_divider
    mae_score = mae_val / mae_divider
    model_score = round(4 * (mape_val / 100) + 3 * rmse_score + 3 * mae_score, 1)

    # Add metrics to the prediction DataFrame
    pred_df['mape'] = round(mape_val,2)
    pred_df['rmse'] = round(rmse_val)
    pred_df['mae'] = round(mae_val)
    pred_df['model_score'] = model_score

    # Calculate growth metrics for the prediction DataFrame
    value_col_name = pred_df.columns[0]
    pred_df['yoy_growth'] = round(pred_df[value_col_name].pct_change(12) * 100, 2) / 100
    pred_df['mom_growth'] = round(pred_df[value_col_name].pct_change(1) * 100,2) / 100

    # Convert actual training data to DataFrame and add identification and metrics columns
    actual_df = train.pd_dataframe()
    actual_df['type'] = type_name
    actual_df['model'] = 'Actual'
    actual_df['mape'] = np.nan
    actual_df['rmse'] = np.nan
    actual_df['mae'] = np.nan
    actual_df['model_score'] = np.nan

    # Calculate growth metrics for the actual data
    actual_df['yoy_growth'] = round(actual_df[value_col_name].pct_change(12) * 100, 2) / 100
    actual_df['mom_growth'] = round(actual_df[value_col_name].pct_change(1) * 100,2) / 100

    # Combine actual and predicted data into one DataFrame
    combined_df = pd.concat([actual_df, pred_df])
    combined_df = combined_df[[combined_df.columns[1], combined_df.columns[2], combined_df.columns[0],
                                combined_df.columns[3], combined_df.columns[4], combined_df.columns[5],
                                combined_df.columns[6], combined_df.columns[7], combined_df.columns[8]]]
    combined_df = combined_df.reset_index().rename(columns={'index': 'date'})

    return combined_df

def out_dim_date(df):
    """
    Generate a DataFrame with date breakdown components.

    Parameters
    ----------
    df : DataFrame
        The DataFrame containing a 'date' column to be broken down.

    Description
    -----------
    This function takes a DataFrame with a 'date' column and breaks it down into 
    components like year, quarter, month, month name, day, and day name. It returns 
    a DataFrame with these additional columns for more detailed date analysis.

    Returns
    -------
    dim_date : DataFrame
        A DataFrame with the original 'date' column and additional date breakdown columns.

    """
    # Create a new DataFrame from the input DataFrame
    dim_date = pd.DataFrame(df)

    # Extract and add date components to the DataFrame
    dim_date['year'] = dim_date['date'].dt.year
    dim_date['quarter'] = dim_date['date'].dt.quarter
    dim_date['month'] = dim_date['date'].dt.month
    dim_date['month_name'] = dim_date['date'].dt.month_name()
    dim_date['day'] = dim_date['date'].dt.dayofweek
    dim_date['day_name'] = dim_date['date'].dt.day_name()

    # Remove duplicate rows based on the date breakdown
    dim_date = dim_date.drop_duplicates()

    return dim_date

def out_dim_model(df, model_col):
    """
    Create a dimension DataFrame for models with unique identifiers and categories.

    Parameters
    ----------
    df : DataFrame
        The DataFrame containing model data.
    model_col : str
        The column name in df that contains the model names.

    Description
    -----------
    This function processes a DataFrame to create a dimension table for different models. 
    It assigns unique identifiers to each model and categorizes them. The function also 
    sets these identifiers as the index of the new DataFrame.

    Returns
    -------
    dim_model : DataFrame
        A dimension DataFrame for models with unique identifiers and categories.

    """
    # Create a DataFrame for models
    dim_model = pd.DataFrame(df)

    # Assign unique identifiers to each model
    dim_model['model_id'] = dim_model[model_col].apply(lambda x: 0 if x == 'Actual'
                                                                    else 1 if x == 'Auto-ARIMA'
                                                                    else 2 if x == 'Prophet'
                                                                    else 3 if x == 'Exponential Smoothing'
                                                                    else 4 if x == 'Bayesian Ridge'
                                                                    else 5 if x == 'LSTM'
                                                                    else 'NA')

    # Categorize each model
    dim_model['model_category'] = dim_model[model_col].apply(lambda x:'NA' if x == 'Actual'
                                                                            else 'Statistical' if x in ['Auto-ARIMA', 'Prophet', 'Exponential Smoothing']
                                                                            else 'ML' if x in ['Bayesian Ridge', 'LSTM'] 
                                                                            else 'NA')

    # Reorder the columns
    dim_model = dim_model[[dim_model.columns[1], dim_model.columns[0], dim_model.columns[2]]]

    # Remove duplicate rows
    dim_model = dim_model.drop_duplicates()

    # Set 'model_id' as the index
    dim_model = dim_model.set_index('model_id')

    return dim_model

def out_dim_type(df, type_col):
    """
    Create a dimension DataFrame for data types with unique identifiers and categories.

    Parameters
    ----------
    df : DataFrame
        The DataFrame containing type data.
    type_col : str
        The column name in df that contains the type names.

    Description
    -----------
    This function processes a DataFrame to create a dimension table for different types. 
    It assigns unique identifiers to each type and categorizes them based on their characteristics. 
    The function also sets these identifiers as the index of the new DataFrame.

    Returns
    -------
    dim_type : DataFrame
        A dimension DataFrame for types with unique identifiers and categories.

    """
    # Create a DataFrame for types
    dim_type = pd.DataFrame(df)

    # Assign unique identifiers to each type
    dim_type['type_id'] = dim_type[type_col].apply(lambda x: 11 if x == 'Debit'
                                                                else 12 if x == 'Credit'
                                                                else 13 if x == 'UE Bank' 
                                                                else 23 if x == 'UE Non-Bank' 
                                                                else 24 if x == 'QRIS' 
                                                                else 21 if x == 'Digital Banking' 
                                                                else 1 if x == 'Google Trends' 
                                                                else 2 if x == 'Electronic Transaction Nom'
                                                                else 3 if x == 'Combined' 
                                                                else 'NA')

    # Categorize each type
    dim_type['type_category'] = dim_type[type_col].apply(lambda x: 'Card' if x in ['Debit', 'Credit', 'UE Bank']
                                                                            else 'Non-card' if x in ['UE Non-Bank', 'QRIS', 'Digital Banking']
                                                                            else 'External' if x == 'Google Trends'
                                                                            else 'Internal' if x == 'Electronic Transaction Nom'
                                                                            else 'Combination' if x == 'Combined' 
                                                                            else 'NA')

    # Reorder the columns
    dim_type = dim_type[[dim_type.columns[1], dim_type.columns[0], dim_type.columns[2]]]

    # Remove duplicate rows
    dim_type = dim_type.drop_duplicates()

    # Set 'type_id' as the index
    dim_type = dim_type.set_index('type_id')

    return dim_type

def out_dim_breakdown(df, breakdown_col):
    """
    Create a dimension DataFrame for transaction breakdown types with unique identifiers.

    Parameters
    ----------
    df : DataFrame
        The DataFrame containing breakdown data.
    breakdown_col : str
        The column name in df that contains the breakdown types.

    Description
    -----------
    This function processes a DataFrame to create a dimension table for different transaction breakdown types. 
    It assigns unique identifiers to each type. The function also sets these identifiers as the index of the new DataFrame.

    Returns
    -------
    dim_breakdown : DataFrame
        A dimension DataFrame for breakdown types with unique identifiers.

    """
    # Create a DataFrame for breakdown types
    dim_breakdown = pd.DataFrame(df[breakdown_col])

    # Assign unique identifiers to each breakdown type
    dim_breakdown['breakdown_id'] = dim_breakdown[breakdown_col].apply(lambda x: 1 if x == 'Tunai'
                                                                                    else 2 if x == 'Non Tunai'
                                                                                    else 3 if x == 'Belanja' 
                                                                                    else 4 if x == 'Setor Tunai'
                                                                                    else 5 if x == 'Pembayaran'
                                                                                    else 6 if x == 'Transfer Intra' 
                                                                                    else 7 if x == 'Transfer Inter' 
                                                                                    else 8 if x == 'Transaksi Online' 
                                                                                    else 9 if x == 'Transfer' 
                                                                                    else 10 if x == 'Cash Advance' 
                                                                                    else 11 if x == 'Bill Payment' 
                                                                                    else 12 if x == 'Initial' 
                                                                                    else 13 if x == 'Transfer Rekening' 
                                                                                    else 14 if x == 'Transfer Pemerintah' 
                                                                                    else 15 if x == 'Top Up' 
                                                                                    else 16 if x == 'Redeem' 
                                                                                    else 17 if x == 'SMS/Mobile Banking' 
                                                                                    else 18 if x == 'Internet Banking' 
                                                                                    else 'NA')

    # Reorder the columns for better organization
    dim_breakdown = dim_breakdown[[dim_breakdown.columns[1], dim_breakdown.columns[0]]]

    # Remove any duplicate rows to ensure uniqueness
    dim_breakdown = dim_breakdown.drop_duplicates()

    # Set 'breakdown_id' as the index for easy reference
    dim_breakdown = dim_breakdown.set_index('breakdown_id')

    return dim_breakdown

def out_dim_cov_breakdown(df, cov_breakdown_col):
    """
    Create a dimension DataFrame for covariate breakdown types with unique identifiers.

    Parameters
    ----------
    df : DataFrame
        The DataFrame containing covariate breakdown data.
    cov_breakdown_col : str
        The column name in df that contains the covariate breakdown types.

    Description
    -----------
    This function processes a DataFrame to create a dimension table for different covariate breakdown types. 
    It assigns unique identifiers to each type and sets these identifiers as the index of the new DataFrame.

    Returns
    -------
    dim_cov_breakdown : DataFrame
        A dimension DataFrame for covariate breakdown types with unique identifiers.

    """
    # Create dim_cov_breakdown DataFrame from df[cov_breakdown_col] column
    dim_cov_breakdown = pd.DataFrame(df[cov_breakdown_col])

    # Create new column 'cov_breakdown_id' and assign each type with a unique number
    dim_cov_breakdown['cov_breakdown_id'] =  dim_cov_breakdown[cov_breakdown_col].apply(lambda x: 1 if x == 'motorcycle'
                                                                                                        else 2 if x == 'auto insurance'
                                                                                                        else 3 if x == 'vehicle wheels & tires'
                                                                                                        else 4 if x == 'vehicle maintanance'
                                                                                                        else 5 if x == 'microcars & city cars'
                                                                                                        else 6 if x == 'motor vehicle & parts'
                                                                                                        else 7 if x == 'home insurance'
                                                                                                        else 8 if x == 'home furnishing'
                                                                                                        else 9 if x == 'homemaking & interior design'
                                                                                                        else 10 if x == 'home improvement'
                                                                                                        else 11 if x == 'bed & bath'
                                                                                                        else 12 if x == 'home appliances'
                                                                                                        else 13 if x == 'furnishing & durable household equipment'
                                                                                                        else 14 if x == 'movies'
                                                                                                        else 15 if x == 'comics & animation'
                                                                                                        else 16 if x == 'TV & video'
                                                                                                        else 17 if x == 'broadway & musical theater'
                                                                                                        else 18 if x == 'computer & video games'
                                                                                                        else 19 if x == 'recreational goods'
                                                                                                        else 20 if x == 'mobile phones'
                                                                                                        else 21 if x == 'e-books'
                                                                                                        else 22 if x == 'camera & photo equipment'  
                                                                                                        else 23 if x == 'audio equipment'
                                                                                                        else 24 if x == 'magazines'
                                                                                                        else 25 if x == 'other durable goods'
                                                                                                        else 26 if x == 'non alcoholic beverages'
                                                                                                        else 27 if x == 'candy & sweets'
                                                                                                        else 28 if x == 'cooking & recipe'
                                                                                                        else 29 if x == 'grocery & food retailers'
                                                                                                        else 30 if x == 'restaurant'
                                                                                                        else 31 if x == 'food and beverages'
                                                                                                        else 32 if x == "men's clothing"
                                                                                                        else 33 if x == "women's clothing"
                                                                                                        else 34 if x == "children's clothing"
                                                                                                        else 35 if x == 'casual apparel'
                                                                                                        else 36 if x == 'athletic apparel'
                                                                                                        else 37 if x == 'footwear'
                                                                                                        else 38 if x == 'clothing & footwear'
                                                                                                        else 39 if x == 'electricity'
                                                                                                        else 40 if x == 'oil & gas'
                                                                                                        else 41 if x == 'nuclear energy'
                                                                                                        else 42 if x == 'waste management'
                                                                                                        else 43 if x == 'renewable & alternative energy'
                                                                                                        else 44 if x == 'gasoline and energy goods'
                                                                                                        else 45 if x == 'medications'
                                                                                                        else 46 if x == 'face & body care'
                                                                                                        else 47 if x == 'hair products'
                                                                                                        else 48 if x == 'cleaning agents'
                                                                                                        else 49 if x == 'tobbaco'
                                                                                                        else 50 if x == 'other nondurable goods'
                                                                                                        else 51 if x == 'qris nominal'
                                                                                                        else 52 if x == 'ue_nonbank nominal'
                                                                                                        else 53 if x == 'ue_bank nominal'
                                                                                                        else 54 if x == 'credit nominal'
                                                                                                        else 55 if x == 'debit nominal'
                                                                                                        else 56 if x == 'digital_banking nominal'
                                                                                                        else 'NA')
    
    # Reorder columns for the  dim_cov_breakdown DataFrame
    dim_cov_breakdown =   dim_cov_breakdown[[dim_cov_breakdown.columns[1],  dim_cov_breakdown.columns[0]]]

    # Remove duplicate rows in  dim_cov_breakdown DataFrame
    dim_cov_breakdown =  dim_cov_breakdown.drop_duplicates()

    # Set type_id as index column
    dim_cov_breakdown =  dim_cov_breakdown.set_index('cov_breakdown_id')

    return  dim_cov_breakdown

def out_fct_forecast(df, fct_col_id, type_col, model_col, date_col='date'):
    """
    Create a fact table for forecast data with identifiers for type and model.

    Parameters
    ----------
    df : DataFrame
        The DataFrame containing forecast data.
    fct_col_id : str
        The column name to be used as the primary key in the fact table.
    type_col : str
        The column name in df that contains the type of data.
    model_col : str
        The column name in df that contains the model names.
    date_col : str, optional
        The column name in df that contains the date. Default is 'date'.

    Description
    -----------
    This function processes a DataFrame to create a fact table for forecast data. 
    It assigns unique identifiers for type and model and creates a primary key 
    for the fact table. The function also filters out rows where the model is 'Actual'.

    Returns
    -------
    DataFrame
        A fact table for forecast data with identifiers and primary key.

    """
     # Create new column 'type_id' and assign each type with a unique number
    df['type_id'] = df[type_col].apply(lambda x: 11 if x == 'Debit'
                                                     else 12 if x == 'Credit'
                                                     else 13 if x == 'UE Bank'
                                                     else 23 if x == 'UE Non-Bank'
                                                     else 24 if x == 'QRIS'
                                                     else 21 if x == 'Digital Banking'
                                                     else 1 if x == 'Google Trends'
                                                     else 2 if x == 'Electronic Transaction Nom'
                                                     else 3 if x == 'Combined'
                                                     else 'NA')
    
    # Create new column 'model_id' and assign each model with a unique number
    df['model_id'] = df[model_col].apply(lambda x: 0 if x == 'Actual' 
                                                        else 1 if x == 'Auto-ARIMA' 
                                                        else 2 if x == 'Prophet' 
                                                        else 3 if x == 'Exponential Smoothing'
                                                        else 4 if x == 'Bayesian Ridge' 
                                                        else 5 if x == 'LSTM'
                                                        else 'NA')
    
    # Create primary key for the fact table
    df[fct_col_id] = df[date_col].astype(str) + '_' + df['type_id'].astype(str)

    # Set the primary key as the first column
    df = df[[df.columns[-1]] + list(df.columns[:-1])]

    # Select only rows with model != 'Actual'
    df = df[df[model_col] != 'Actual']

    # Drop unnecessary columns
    df = df.drop(columns=[model_col, type_col])
     
    # Drop duplicate rows
    df = df.drop_duplicates()

    return df

def out_fct_actual(df,fct_col_id, model_col, type_col, date_col='date'):
    """
    Create a fact table for actual data with identifiers for type and model.

    Parameters
    ----------
    df : DataFrame
        The DataFrame containing actual data.
    fct_col_id : str
        The column name to be used as the primary key in the fact table.
    model_col : str
        The column name in df that contains the model names.
    type_col : str
        The column name in df that contains the type of data.
    date_col : str, optional
        The column name in df that contains the date. Default is 'date'.

    Description
    -----------
    This function processes a DataFrame to create a fact table for actual data. 
    It assigns unique identifiers for type and model and creates a primary key 
    for the fact table. The function filters to include only rows where the model is 'Actual'.

    Returns
    -------
    DataFrame
        A fact table for actual data with identifiers and primary key.

    """
    # Create new column 'type_id' and assign each type with a unique number
    df['type_id'] = df[type_col].apply(lambda x: 11 if x == 'Debit'
                                                     else 12 if x == 'Credit'
                                                     else 13 if x == 'UE Bank'
                                                     else 23 if x == 'UE Non-Bank'
                                                     else 24 if x == 'QRIS'
                                                     else 21 if x == 'Digital Banking'
                                                     else 1 if x == 'Google Trends'
                                                     else 2 if x == 'Electronic Transaction Nom'
                                                     else 3 if x == 'Combined'
                                                     else 'NA')
    
    # Create new column 'model_id' and assign each model with a unique number
    df['model_id'] = df[model_col].apply(lambda x: 0 if x == 'Actual' 
                                                        else 1 if x == 'Auto-ARIMA' 
                                                        else 2 if x == 'Prophet' 
                                                        else 3 if x == 'Exponential Smoothing'
                                                        else 4 if x == 'Bayesian Ridge' 
                                                        else 5 if x == 'LSTM'
                                                        else 'NA')
    
    # Create primary key for the fact table
    df[fct_col_id] = df[date_col].astype(str) + '_' + df['type_id'].astype(str)

    # Set the primary key as the first column
    df = df[[df.columns[-1]] + list(df.columns[:-1])]

    # Select only rows with model != 'Actual'
    df = df[df[model_col] == 'Actual']

    # Drop unnecessary columns
    df = df.drop(columns=[model_col, type_col])

    # Drop duplicate rows
    df = df.drop_duplicates()

    return df

def out_fct_breakdown(df, fct_col_id, breakdown_col, type_col, date_col='date'):
    """
    Create a fact table for breakdown data with unique identifiers for each breakdown type.

    Parameters
    ----------
    df : DataFrame
        The DataFrame containing breakdown data.
    fct_col_id : str
        The column name to be used as the primary key in the fact table.
    breakdown_col : str
        The column name in df that contains the breakdown types.
    type_col : str
        The column name in df that contains the type of data.
    date_col : str, optional
        The column name in df that contains the date. Default is 'date'.

    Description
    -----------
    This function processes a DataFrame to create a fact table for breakdown data. 
    It assigns unique identifiers for each breakdown type and creates a primary key 
    for the fact table.

    Returns
    -------
    DataFrame
        A fact table for breakdown data with unique identifiers and primary key.

    """
    # Create new column 'breakdown_id' and assign each type with a unique number
    df['breakdown_id'] =  df[breakdown_col].apply(lambda x: 1 if x == 'Tunai'
                                                                else 2 if x == 'Non Tunai' 
                                                                else 3 if x == 'Belanja' 
                                                                else 4 if x == 'Setor Tunai'
                                                                else 5 if x == 'Pembayaran'
                                                                else 6 if x == 'Transfer Intra'
                                                                else 7 if x == 'Transfer Inter'
                                                                else 8 if x == 'Transaksi Online'
                                                                else 9 if x == 'Transfer'
                                                                else 10 if x == 'Cash Advance'
                                                                else 11 if x == 'Bill Payment'
                                                                else 12 if x == 'Initial'
                                                                else 13 if x == 'Transfer Rekening'
                                                                else 14 if x == 'Transfer Pemerintah'
                                                                else 15 if x == 'Top Up'
                                                                else 16 if x == 'Redeem'
                                                                else 17 if x == 'SMS/Mobile Banking'
                                                                else 18 if x == 'Internet Banking'
                                                                else 'NA')
    
     # Create new column 'type_id' and assign each type with a unique number
    df['type_id'] = df[type_col].apply(lambda x: 11 if x == 'Debit'
                                                     else 12 if x == 'Credit'
                                                     else 13 if x == 'UE Bank'
                                                     else 23 if x == 'UE Non-Bank'
                                                     else 24 if x == 'QRIS'
                                                     else 21 if x == 'Digital Banking'
                                                     else 1 if x == 'Google Trends'
                                                     else 2 if x == 'Electronic Transaction Nom'
                                                     else 3 if x == 'Combined'
                                                     else 'NA')
    
    # Create primary key for the fact table
    df[fct_col_id] = df[date_col].astype(str) + '_' + df['type_id'].astype(str)

    # Set the primary key as the first column
    df = df[[df.columns[-1]] + list(df.columns[:-1])]

    # Drop unnecessary columns
    df = df.drop(columns=[breakdown_col, type_col])

    # Drop duplicate rows
    df = df.drop_duplicates()

    return df

def out_fct_cov_breakdown(df, fct_col_id, type_col, model_col, cov_breakdown_col, date_col='date'):
    """
    Create a fact table for covariate breakdown data with unique identifiers.

    Parameters
    ----------
    df : DataFrame
        The DataFrame containing covariate breakdown data.
    fct_col_id : str
        The column name to be used as the primary key in the fact table.
    type_col : str
        The column name in df that contains the type of data.
    model_col : str
        The column name in df that contains the model names.
    cov_breakdown_col : str
        The column name in df that contains covariate breakdown types.
    date_col : str, optional
        The column name in df that contains the date. Default is 'date'.

    Description
    -----------
    This function processes a DataFrame to create a fact table for covariate breakdown data. 
    It assigns unique identifiers for each covariate breakdown type, model, and data type, 
    and then creates a primary key for the fact table.

    Returns
    -------
    DataFrame
        A fact table for covariate breakdown data with identifiers and primary key.

    """
    # Create new column 'type_id' and assign each type with a unique number
    df['type_id'] = df[type_col].apply(lambda x: 11 if x == 'Debit'
                                                     else 12 if x == 'Credit'
                                                     else 13 if x == 'UE Bank'
                                                     else 23 if x == 'UE Non-Bank'
                                                     else 24 if x == 'QRIS'
                                                     else 21 if x == 'Digital Banking'
                                                     else 1 if x == 'Google Trends'
                                                     else 2 if x == 'Electronic Transaction Nom'
                                                     else 3 if x == 'Combined'
                                                     else 'NA')
    
     # Create new column 'model_id' and assign each model with a unique number
    df['model_id'] = df[model_col].apply(lambda x: 0 if x == 'Actual' 
                                                        else 1 if x == 'Auto-ARIMA' 
                                                        else 2 if x == 'Prophet' 
                                                        else 3 if x == 'Exponential Smoothing'
                                                        else 4 if x == 'Bayesian Ridge' 
                                                        else 5 if x == 'LSTM'
                                                        else 'NA')
    
    
    # Create new column 'cov_breakdown_id' and assign each type with a unique number
    df['cov_breakdown_id'] =  df[cov_breakdown_col].apply(lambda x: 1 if x == 'motorcycle'
                                                                        else 2 if x == 'auto insurance'
                                                                        else 3 if x == 'vehicle wheels & tires'
                                                                        else 4 if x == 'vehicle maintanance'
                                                                        else 5 if x == 'microcars & city cars'
                                                                        else 6 if x == 'motor vehicle & parts'
                                                                        else 7 if x == 'home insurance'
                                                                        else 8 if x == 'home furnishing'
                                                                        else 9 if x == 'homemaking & interior design'
                                                                        else 10 if x == 'home improvement'
                                                                        else 11 if x == 'bed & bath'
                                                                        else 12 if x == 'home appliances'
                                                                        else 13 if x == 'furnishing & durable household equipment'
                                                                        else 14 if x == 'movies'
                                                                        else 15 if x == 'comics & animation'
                                                                        else 16 if x == 'TV & video'
                                                                        else 17 if x == 'broadway & musical theater'
                                                                        else 18 if x == 'computer & video games'
                                                                        else 19 if x == 'recreational goods'
                                                                        else 20 if x == 'mobile phones'
                                                                        else 21 if x == 'e-books'
                                                                        else 22 if x == 'camera & photo equipment'  
                                                                        else 23 if x == 'audio equipment'
                                                                        else 24 if x == 'magazines'
                                                                        else 25 if x == 'other durable goods'
                                                                        else 26 if x == 'non alcoholic beverages'
                                                                        else 27 if x == 'candy & sweets'
                                                                        else 28 if x == 'cooking & recipe'
                                                                        else 29 if x == 'grocery & food retailers'
                                                                        else 30 if x == 'restaurant'
                                                                        else 31 if x == 'food and beverages'
                                                                        else 32 if x == "men's clothing"
                                                                        else 33 if x == "women's clothing"
                                                                        else 34 if x == "children's clothing"
                                                                        else 35 if x == 'casual apparel'
                                                                        else 36 if x == 'athletic apparel'
                                                                        else 37 if x == 'footwear'
                                                                        else 38 if x == 'clothing & footwear'
                                                                        else 39 if x == 'electricity'
                                                                        else 40 if x == 'oil & gas'
                                                                        else 41 if x == 'nuclear energy'
                                                                        else 42 if x == 'waste management'
                                                                        else 43 if x == 'renewable & alternative energy'
                                                                        else 44 if x == 'gasoline and energy goods'
                                                                        else 45 if x == 'medications'
                                                                        else 46 if x == 'face & body care'
                                                                        else 47 if x == 'hair products'
                                                                        else 48 if x == 'cleaning agents'
                                                                        else 49 if x == 'tobbaco'
                                                                        else 50 if x == 'other nondurable goods'
                                                                        else 51 if x == 'qris nominal'
                                                                        else 52 if x == 'ue_nonbank nominal'
                                                                        else 53 if x == 'ue_bank nominal'
                                                                        else 54 if x == 'credit nominal'
                                                                        else 55 if x == 'debit nominal'
                                                                        else 56 if x == 'digital_banking nominal'
                                                                        else 'NA')
    

    
    # Create primary key for the fact table
    df[fct_col_id] = df[date_col].astype(str) + '_' + df['cov_breakdown_id'].astype(str)

    # Set the primary key as the first column
    df = df[[df.columns[-1]] + list(df.columns[:-1])]

    # Drop unnecessary columns
    df = df.drop(columns=[model_col, type_col, cov_breakdown_col])

    # Drop duplicate rows
    df = df.drop_duplicates()

    return df