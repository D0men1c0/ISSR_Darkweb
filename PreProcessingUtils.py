import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import seaborn as sns
import tldextract
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction import FeatureHasher
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Input, Dense, Flatten
from tensorflow.keras.models import Model
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

def get_dummies(df, feature):
    """
    This function is used to convert the categorical feature into one-hot encoding.
    :param df: The dataframe containing the feature.
    :param feature: The feature to be converted.
    :return: The dataframe with the one-hot encoding of the feature.
    """
    return pd.get_dummies(df, columns=[feature])

def standardize(df, feature):
    """
    This function is used to standardize the feature.
    :param df: The dataframe containing the feature.
    :param feature: The feature to be standardized.
    :return: The dataframe with the standardized feature.
    """
    scaler = StandardScaler()
    df[feature] = scaler.fit_transform(df[[feature]])
    return df

def normalize(df, feature):
    """
    This function is used to normalize the feature.
    :param df: The dataframe containing the feature.
    :param feature: The feature to be normalized.
    :return: The dataframe with the normalized feature.
    """
    df[feature] = (df[feature] - df[feature].min()) / (df[feature].max() - df[feature].min())
    return df

def binarize(df, feature, threshold):
    """
    This function is used to binarize the feature.
    :param df: The dataframe containing the feature.
    :param feature: The feature to be binarized.
    :param threshold: The threshold to binarize the feature.
    :return: The dataframe with the binarized feature.
    """
    df[feature] = np.where(df[feature] > threshold, 1, 0)
    return df

def discretize(df, feature, bins):
    """
    This function is used to discretize the feature.
    :param df: The dataframe containing the feature.
    :param feature: The feature to be discretized.
    :param bins: The number of bins to discretize the feature.
    :return: The dataframe with the discretized feature.
    """
    df[feature] = pd.cut(df[feature], bins, labels=False)
    return df

def remove_outliers(df, feature):
    """
    This function is used to remove the outliers from the feature.
    :param df: The dataframe containing the feature.
    :param feature: The feature to remove the outliers.
    :return: The dataframe with the outliers removed from the feature.
    """
    q1 = df[feature].quantile(0.25)
    q3 = df[feature].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    df = df[(df[feature] > lower_bound) & (df[feature] < upper_bound)]
    return df

def encode_date_numerically(df, feature):
    """
    This function is used to encode the date feature numerically.
    :param df: The dataframe containing the feature.
    :param feature: The feature to be encoded.
    :return: The dataframe with the date feature encoded numerically.
    """
    df[feature] = pd.to_datetime(df[feature])
    df[feature] = df[feature].map(lambda x: 10000*x.year + 100*x.month + x.day)
    return df

def encode_date_cyclically(df, feature):
    """
    This function is used to encode the date feature cyclically.
    :param df: The dataframe containing the feature.
    :param feature: The feature to be encoded.
    :return: The dataframe with the date feature encoded cyclically.
    """
    df[feature + '_month_sin'] = np.sin(2 * np.pi * df[feature].dt.month / 12)
    df[feature + '_month_cos'] = np.cos(2 * np.pi * df[feature].dt.month / 12)
    df[feature + '_year_sin'] = np.sin(2 * np.pi * df[feature].dt.year)
    df[feature + '_year_cos'] = np.cos(2 * np.pi * df[feature].dt.year)
    return df

def preprocess_url(df, feature):
    """
    This function is used to preprocess the URL feature.
    :param df: The dataframe containing the feature.
    :param feature: The feature to be preprocessed.
    :return: The dataframe with the preprocessed URL feature.
    """
    df[feature] = df[feature].str.replace('http://', '')
    df[feature] = df[feature].str.replace('https://', '')
    df[feature] = df[feature].str.replace('www.', '')
    df[feature] = df[feature].str.split('.').str[0]
    return df
