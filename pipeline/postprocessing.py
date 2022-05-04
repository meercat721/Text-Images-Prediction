from pathlib import Path

from matplotlib import pyplot as plt
from pandas import DataFrame


def save_top_100(results: DataFrame):
    """
    Saves the top 100 results to a csv (with article id, rank)
    :param results: the computed results
    :return:
    """
    pass 

def plot_loss(trained_model):
    """
    Plots the loss of a trained model.
    :param trained_model: the model in question
    :return:
    """
    plt.plot(trained_model.history["loss"], label="Training loss")
    plt.plot(trained_model.history["val_loss"], label="Validation Loss")
    plt.ylim([0, 0.01])
    plt.xlabel("Epoch")
    plt.ylabel("CTR Error")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_prediction_scatter(label_values, predictions, limits):
    """
    Scatter plot of the predicted vs. actual vales
    :param label_values: the actual values of the label to predict
    :param predictions: the model prediction values
    :param limits: graph limits
    :return:
    """
    plt.axes(aspect="equal")
    plt.scatter(label_values, predictions)
    plt.xlabel("True ctr values")
    plt.ylabel("CTR Predictions")
    plt.xlim(limits)
    plt.ylim(limits)
    plt.plot(limits, limits)
    plt.show()


def plot_error_distribution(predictions, label_values, bin_count, limits):
    """
    Plot the distribution of the prediction error in a histogram
    :param predictions: the prediction values
    :param label_values: the label values
    :param bin_count: amount of bins (columns) in the histogram
    :param limits: histogram limits
    :return:
    """
    plt.hist(predictions - label_values, bins=bin_count)
    plt.xlim(limits)
    plt.xlabel("CTR prediction error")
    plt.ylabel("Count")
    plt.show()


def save_processed(df: DataFrame, filename: str):
    """
    Saves a dataframe to a csv
    :param df: the dataframe
    :param filename: name of the file
    :return:
    """
    df.to_csv(f"./processed_data/{filename}.tsv", sep="\t")


def check_processed(filename: str) -> bool:
    """
    check if a processed csv file exists
    :param filename: the file name
    :return: true or false depending on existing file
    """
    file = Path(f"./processed_data/{filename}")
    return file.is_file()
