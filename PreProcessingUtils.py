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
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import langid
import time
import math
from deep_translator import GoogleTranslator

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

def convert_date(df, feature):
    """
    This function is used to convert the date feature to a specific format.
    :param df: The dataframe containing the feature.
    :param feature: The feature to be converted.
    :return: The dataframe with the date feature converted to a specific format.
    """
    try:
        df[feature] = pd.to_datetime(df[feature], format='%Y-%m-%d %H:%M:%S.%f').dt.strftime('%Y-%m-%d')
    except:
        df[feature] = pd.to_datetime(df[feature], format='%Y-%m-%d %H:%M:%S').dt.strftime('%Y-%m-%d')
    return df

def extract_date_features(df, feature, fill='1900-01-01'):
    """
    This function is used to extract the date features from the date feature.
    :param df: The dataframe containing the feature.
    :param feature: The feature to extract the date features.
    :param fill: The value to fill the missing values in the feature.
    :return: The dataframe with the date features extracted from the date feature.
    """
    df[feature] = df[feature].fillna(fill)
    df[feature + '_year'] = (pd.to_datetime(df[feature]).dt.year).astype(str)
    df[feature + '_month'] = (pd.to_datetime(df[feature]).dt.month).astype(str)
    df[feature + '_day'] = (pd.to_datetime(df[feature]).dt.day).astype(str)
    df[feature + '_dayofweek'] = (pd.to_datetime(df[feature]).dt.dayofweek).astype(str)
    df[feature + '_is_weekend'] = pd.to_datetime(df[feature]).dt.dayofweek.isin([5, 6]).astype(int)
    return df

def encode_date_cyclically(df, feature):
    """
    This function is used to encode the date feature cyclically.
    :param df: The dataframe containing the feature.
    :param feature: The feature to be encoded.
    :return: The dataframe with the date feature encoded cyclically.
    """
    df[feature] = pd.to_datetime(df[feature], format='%Y-%m')
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

def minmax_scale_column(df, column_name):
    """
    This function is used to min-max scale the column.
    :param df: The dataframe containing the column.
    :param column_name: The column to be min-max scaled.
    :return: The dataframe with the min-max scaled column.
    """
    scaler = MinMaxScaler()
    df[column_name] = scaler.fit_transform(df[[column_name]])
    
    return df


def detect_languages(df, feature):
    languages = []
    for text in tqdm(df[feature], desc='Detecting languages'):
        language, _ = langid.classify(text)
        languages.append(language)
    df['language'] = languages
    return df

def translate_texts_in_dataframe(df, text_feature, lang_feature, batch_size=1000, output_file=r'C:\Users\dommy\OneDrive\Documenti\GSoC - Final\GSoC\CleanedCSV\threads_preprocessed_translated.csv'):
    """
    This function is used to translate the texts in the dataframe to English.
    :param df: The dataframe containing the texts.
    :param text_feature: The feature containing the texts.
    :param lang_feature: The feature containing the languages of the texts.
    :param batch_size: The batch size for translation.
    :param output_file: The output file to save the translated dataframe.
    :return: The dataframe with the translated texts.
    """
    non_english_df = df[df[lang_feature] != 'en']
    num_batches = math.ceil(len(non_english_df) / batch_size)
    with tqdm(total=len(non_english_df), desc="Translating texts") as pbar:
        for i in range(num_batches):
            batch_start = i * batch_size
            batch_end = min((i + 1) * batch_size, len(non_english_df))
            batch_df = non_english_df.iloc[batch_start:batch_end]
            batch_texts = batch_df[text_feature].tolist()
            batch_languages = batch_df[lang_feature].tolist()
            try:
                translated_batch = GoogleTranslator(source=batch_languages, target='en').translate_batch(batch_texts)
            except:
                try:
                    translated_batch = GoogleTranslator(source='auto', target='en').translate_batch(batch_texts)
                except:
                    return df
            df.loc[batch_df.index, text_feature] = translated_batch
            df.loc[batch_df.index, lang_feature] = 'en'
            pbar.update(len(batch_df))
            time.sleep(0.1)
            df.to_csv(output_file, index=False)
    return df