# Import Libraries
import matplotlib.pyplot as plt
import missingno as mso
import seaborn as sns
import math
from darts.metrics import mape, rmse, mae
from darts.dataprocessing.transformers import Scaler
scaler = Scaler()

# Libraries Settings
sns.set_style('whitegrid')
plt.rcParams['figure.dpi']=100

def plot_distribution(df, type):
    """
    Plot the distribution of each numerical column in a DataFrame using histograms and box plots.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe containing the data.
    type : str
        A descriptive string used in the plot title to specify the type of data being analyzed (e.g., 'Credit', 'Debit').

    Description
    -----------
    This function iterates through each numerical column in the dataframe and creates two types of plots for each:
    1. A histogram with a Kernel Density Estimate (KDE) to visualize the distribution of the data.
    2. A box plot to visualize the quartiles and outliers of the data.
    These plots provide a comprehensive view of the data distribution, helping in identifying patterns, skewness, and outliers.
    
    """
    # Exception handling for non-numerical columns
    try:
        # Iterate over each numerical column in the DataFrame
        for column in df.select_dtypes(exclude='object').columns:
            # Create a figure with two subplots (1 row, 2 columns) and set the figure size
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))

            # Plot a histogram with KDE for the column on the first subplot
            sns.histplot(df[column], kde=True, ax=ax[0], color='#0698DC')

            # Plot a boxplot for the column on the second subplot
            sns.boxplot(df[column], ax=ax[1], color='#05E6FA')

            # Set the title for the figure with the specified type and column name
            fig.suptitle(f"{type} {column} Distribution", fontweight='bold', fontsize='15', fontfamily='sans-serif', color='#100C07')

            # Display the plots
            plt.show()
    except:
        # Print an error message if the column is non-numerical
        print(f"{column} is not a numerical column.")

def plot_missing_values(df, suptitle='Missing Values in each Columns', title=''):
    """
    Plot the count of missing values in each column of a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe containing the data.
    suptitle : str, optional
        The main title for the plot.
    title : str, optional
        The subtitle for the plot.

    Description
    -----------
    This function calculates the number of missing values in each column of the DataFrame and 
    visualizes this count using a bar plot. It helps in identifying columns with significant numbers 
    of missing values, which is important for data cleaning and preprocessing.

    """
    # Print the main title in bold format
    print('\033[1m' + suptitle + '\033[0m')

    # Print the count of missing values for each column in the DataFrame
    print(df.isnull().sum())

    # Create a bar plot showing the count of missing values for each column
    mso.bar(df, fontsize=9, color=['#0698DC'], figsize=(10, 4), sort='descending', labels=True)

    # Set the main title and subtitle for the plot with custom formatting
    plt.suptitle(suptitle, fontweight='heavy', x=0.122, y=1.08, ha='left', fontsize='15', fontfamily='sans-serif', color='#100C07')
    plt.title(title, fontsize='8', fontfamily='sans-serif', loc='left', color='#3E3B39', pad=5)

    # Add a grid to the plot for better readability
    plt.grid(axis='both', alpha=0)

def plot_train_val(train, train_label, val, val_label, suptitle=''):
    """
    Plot training and validation data series on the same plot for comparison.

    Parameters
    ----------
    train : pandas.Series or pandas.DataFrame
        The training data series or dataframe.
    train_label : str
        The label for the training data in the plot.
    val : pandas.Series or pandas.DataFrame
        The validation data series or dataframe.
    val_label : str
        The label for the validation data in the plot.
    suptitle : str, optional
        The main title for the plot.

    Description
    -----------
    This function plots both the training and validation datasets on the same plot to 
    visually compare and analyze them. It is particularly useful for assessing the 
    partitioning of data into training and validation sets in time series analysis.
    
    """
    # Calculate the length of the training and validation datasets
    train_len = len(train)
    val_len = len(val)

    # Initialize a new figure for plotting
    plt.figure()

    # Plot the training data using the provided label
    train.plot(label=train_label)

    # Plot the validation data using the provided label
    val.plot(label=val_label)

    # Set the title for the plot, showing the lengths of the training and validation datasets
    plt.title('Train Length: {:.2f}, Validation Length: {:.2f}'.format(train_len, val_len))

    # Set the super title (main title) for the plot with custom formatting
    plt.suptitle(suptitle, fontweight='heavy', fontsize='12', fontfamily='sans-serif')

def plot_pred(model, len, train, val, suptitle):
    """
    Fit a time series model to the training data, make predictions, and plot these along with the training and validation data. 

    Parameters
    ----------
    model : TimeSeries model
        The time series model to be fitted and used for prediction.
    len : int
        The length of the forecast horizon.
    train : TimeSeries
        The training data series.
    val : TimeSeries
        The validation data series for comparison.
    suptitle : str
        The main title for the plot.

    Description
    -----------
    This function fits the provided time series model on the training data, predicts future values for a specified length, 
    and then plots the training data, validation data, and predictions for visual comparison. It also calculates and displays 
    evaluation metrics (MAPE, RMSE, MAE) and a composite model score.

    """
    # Create exception handling for non TimeSeries inputs
    try:
        # Fit the model with the training data
        model.fit(train)

        # Predict future values for the specified length
        pred = model.predict(len)

        # Plot training, validation, and predicted data on the same plot
        train.plot(label="train")
        val.plot(label="validation")
        pred.plot(label="prediction")

        # Calculate evaluation metrics for the prediction against validation data
        mape_val = mape(pred, val)
        rmse_val = rmse(pred, val) / 1e9  # Normalizing RMSE
        mae_val = mae(pred, val) / 1e9   # Normalizing MAE

        # Normalize RMSE and MAE by dividing by an order of magnitude divider
        rmse_divider = 10 ** round(math.log10(rmse_val))
        mae_divider = 10 ** round(math.log10(mae_val))
        mape_score = mape_val / 100  # Convert MAPE percentage to a decimal
        rmse_score = rmse_val / rmse_divider
        mae_score = mae_val / mae_divider

        # Calculate a composite model score based on the normalized metrics
        model_score = round(4 * mape_score + 3 * rmse_score + 3 * mae_score, 1)

        # Set the title of the plot with the calculated metrics and model score
        plt.title('MAPE: {:.2f}%, RMSE: {:.0f}M, MAE: {:.0f}M, Score: {:.1f}'.format(mape_val, rmse_val, mae_val, model_score), fontsize='9')
        plt.suptitle(suptitle, fontweight='heavy', fontsize='10', fontfamily='sans-serif')

    except:
        # Print an error message if the input is not a TimeSeries
        print("Input is not a TimeSeries.")

def plot_pred_ml(model, len, train, val, cov, suptitle):
    """
    Fit a machine learning model to the training data with covariates, make predictions, 
    and plot these along with the training and validation data.

    Parameters
    ----------
    model : Machine Learning model
        The machine learning model to be fitted and used for prediction.
    len : int
        The length of the forecast horizon.
    train : TimeSeries
        The training data series.
    val : TimeSeries
        The validation data series for comparison.
    cov : TimeSeries
        The covariates data series.
    suptitle : str
        The main title for the plot.

    Description
    -----------
    This function fits the provided machine learning model on the training data with covariates, 
    predicts future values for a specified length, and then plots the training data, validation data, 
    and predictions for visual comparison. It also calculates and displays evaluation metrics (MAPE, RMSE, MAE) 
    and a composite model score.
    
    """
    # Create exception handling for non TimeSeries inputs
    try:
        # Scale the training and covariates data
        scaled_train, scaled_cov = scaler.fit_transform([train, cov])

        # Fit the model with the scaled training data and covariates
        model.fit(scaled_train, future_covariates=scaled_cov)

        # Predict future values using the model and scale back the predictions
        pred_scaled = model.predict(len, future_covariates=scaled_cov)
        pred = scaler.inverse_transform(pred_scaled)

        # Plot the training, validation, and predicted data
        train.plot(label="train")
        val.plot(label="validation")
        pred.plot(label="prediction")

        # Calculate evaluation metrics for the prediction against validation data
        mape_val = mape(pred, val)
        rmse_val = rmse(pred, val) / 1e9  # Normalizing RMSE
        mae_val = mae(pred, val) / 1e9   # Normalizing MAE

        # Normalize the RMSE and MAE by dividing by an order of magnitude divider
        rmse_divider = 10 ** round(math.log10(rmse_val))
        mae_divider = 10 ** round(math.log10(mae_val))
        mape_score = mape_val / 100  # Convert MAPE percentage to a decimal
        rmse_score = rmse_val / rmse_divider
        mae_score = mae_val / mae_divider

        # Calculate a composite model score based on the normalized metrics
        model_score = round(4 * mape_score + 3 * rmse_score + 3 * mae_score, 1)

        # Set the title of the plot with the calculated metrics and model score
        plt.title('MAPE: {:.2f}%, RMSE: {:.2f}M, MAE: {:.2f}M, Score: {:.2f}'.format(mape_val, rmse_val, mae_val, model_score), fontsize='9')
        plt.suptitle(suptitle, fontweight='heavy', fontsize='10', fontfamily='sans-serif')
    except:
        # Print an error message if the input is not a TimeSeries
        print("Input is not a TimeSeries.")

def plot_backtest(model, train, suptitle, start_val=0.8):
    """
    Perform backtesting on a time series model and plot the actual versus backtest predictions.

    Parameters
    ----------
    model : TimeSeries model
        The time series model to perform backtesting on.
    train : TimeSeries
        The training data series used for backtesting.
    suptitle : str
        The main title for the plot.
    start_val : float, optional
        The start point for backtesting as a fraction of the training data.

    Description
    -----------
    This function performs backtesting on the provided time series model using a portion of the training data. 
    It plots the actual data against the backtest predictions and calculates evaluation metrics (MAPE, RMSE, MAE) 
    to assess the model's performance during backtesting.
    
    """
    # Generate backtest predictions using the model
    historical_pred = model.historical_forecasts(train, start=start_val, forecast_horizon=1, verbose=True)
    
    # Plot the actual training data and backtest predictions
    train.plot(label="actual")
    historical_pred.plot(label="backtest prediction")

    # Calculate evaluation metrics for the backtest predictions against the actual training data
    mape_val = mape(historical_pred, train)
    rmse_val = rmse(historical_pred, train) / 1e9  # Normalizing RMSE
    mae_val = mae(historical_pred, train) / 1e9   # Normalizing MAE

    # Normalize RMSE and MAE by dividing by an order of magnitude divider
    rmse_divider = 10 ** round(math.log10(rmse_val))
    mae_divider = 10 ** round(math.log10(mae_val))
    mape_score = mape_val / 100  # Convert MAPE percentage to a decimal
    rmse_score = rmse_val / rmse_divider
    mae_score = mae_val / mae_divider

    # Calculate a composite model score based on the normalized metrics
    model_score = round(4 * mape_score + 3 * rmse_score + 3 * mae_score, 1)

    # Set the title of the plot with the calculated metrics and model score
    plt.title('MAPE: {:.2f}%, RMSE: {:.0f}M, MAE: {:.0f}M, Score: {:.1f}'.format(mape_val, rmse_val, mae_val, model_score), fontsize='9')
    plt.suptitle(suptitle, fontweight='heavy', fontsize='12', fontfamily='sans-serif')

def plot_ml_backtest(model, train, cov, suptitle, start_val=0.8):
    """
    Perform backtesting on a machine learning model using training data and covariates, 
    then plot the actual training data against the backtest predictions.

    Parameters
    ----------
    model : Machine Learning model
        The machine learning model to be used for backtesting.
    train : TimeSeries
        The training data series.
    cov : TimeSeries
        The covariates data series.
    suptitle : str
        The main title for the plot.
    start_val : float, optional
        The start point for backtesting as a fraction of the training data.

    Description
    -----------
    This function performs backtesting on a machine learning model using scaled training data with covariates. 
    It plots the actual training data against the backtest predictions and calculates evaluation metrics 
    (MAPE, RMSE, MAE) to assess the model's performance during backtesting.
    
    """
    # Scale the training and covariates data
    scaled_train, scaled_cov = scaler.fit_transform([train, cov])

    # Generate backtest predictions using the scaled training data and covariates
    historical_pred_scaled = model.historical_forecasts(scaled_train, future_covariates=scaled_cov, start=start_val, forecast_horizon=1, verbose=True)
    
    # Scale back the predictions to their original scale
    historical_pred = scaler.inverse_transform(historical_pred_scaled)

    # Plot the actual training data and the backtest predictions
    train.plot(label="actual")
    historical_pred.plot(label="backtest prediction")

    # Calculate evaluation metrics for the backtest predictions against the actual training data
    mape_val = mape(historical_pred, train)
    rmse_val = rmse(historical_pred, train) / 1e9  # Normalizing RMSE
    mae_val = mae(historical_pred, train) / 1e9   # Normalizing MAE

    # Normalize RMSE and MAE by dividing by an order of magnitude divider
    rmse_divider = 10 ** round(math.log10(rmse_val))
    mae_divider = 10 ** round(math.log10(mae_val))
    mape_score = mape_val / 100  # Convert MAPE percentage to a decimal
    rmse_score = rmse_val / rmse_divider
    mae_score = mae_val / mae_divider

    # Calculate a composite model score based on the normalized metrics
    model_score = round(4 * mape_score + 3 * rmse_score + 3 * mae_score, 1)

    # Set the title of the plot with the calculated metrics and model score
    plt.title('MAPE: {:.2f}%, RMSE: {:.0f}M, MAE: {:.0f}M, Score: {:.1f}'.format(mape_val, rmse_val, mae_val, model_score), fontsize='9')
    plt.suptitle(suptitle, fontweight='heavy', fontsize='10', fontfamily='sans-serif')   

def plot_actual_pred(actual, pred, suptitle):
    """
    Plot the actual and predicted data series on the same plot for comparison.

    Parameters
    ----------
    actual : TimeSeries
        The actual data series.
    pred : TimeSeries
        The predicted data series.
    suptitle : str
        The main title for the plot.

    Description
    -----------
    This function plots both the actual and predicted data series on the same plot, 
    which is useful for visually comparing the model's predictions against the true values. 
    It helps in assessing the accuracy and performance of a predictive model.
    
    """
    # Plot the actual data series with a label
    actual.plot(label="Actual")

    # Plot the predicted data series with a label
    pred.plot(label="Prediction")

    # Set the super title (main title) for the plot with custom formatting
    plt.suptitle(suptitle, fontweight='heavy', fontsize='12', fontfamily='sans-serif')
