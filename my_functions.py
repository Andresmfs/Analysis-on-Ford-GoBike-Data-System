
import matplotlib.pyplot as plt
import numpy as np


def drop_outliers_IQR(data, column_name):
    '''
    This function drops outliers from the values in a column in a dataframe and returns a copy of the dataframe without outliers.
    It drops al values outside the interval (mean - 1.5*IQR, mean + 1.5*IQR)
    Being IQR the interquartile range.
    '''
    # We calculate the quartiles
    Q1 = np.quantile(data[column_name], 0.25)
    Q3 = np.quantile(data[column_name], 0.75)

    # We calculate the interquartile range
    IQR = Q3 - Q1

    # We calculate the mean value
    mean = np.mean(data[column_name])

    # We find and drop outliers
    out_idx = data[(data[column_name] < mean - 1.5 * IQR) |
                   (data[column_name] > mean + 1.5 * IQR)].index
    data_without_outliers = data.drop(index=out_idx)
    return data_without_outliers


def random_sample(data, n, seed=0):
    '''This function takes a random sample from a dataframe
    data: Dataframe, n: size of the sample'''

    assert n < data.shape[0], 'sample size (n) must be smaller than the number of rows of the dataframe'

    np.random.seed(seed)
    sample_idx = np.random.choice(data.shape[0], n, replace=False)
    return data.loc[sample_idx]


def print_percentage(data, column_name: str, order, x_shift=0.2, y_shift=0.02, horizontal=True):
    '''
    This function is to be used below a count plot. 
    It prints the percentage on the bars of vertical countplots
    '''

    val_counts = data[column_name].value_counts().loc[order]
    total_counts = val_counts.sum()

    max_count = val_counts.sort_values(ascending=False).iloc[0]

    # Text labels
    for i, count in enumerate(val_counts):
        pct_string = f'{100*count/total_counts:.1f}%'
        plt.text(i - x_shift, count + y_shift *
                 max_count, pct_string, va='center')
