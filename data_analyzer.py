from re import sub
from matplotlib.pyplot import subplots
import pandas as pd
import os
import urllib
import urllib.request
from urllib.request import urlopen
from word_search import *
from pandas.core.frame import DataFrame
import json

# Sort s1 w/ prices via date

def sort_data_cik(to_modify_csv):
    to_modify_df = pd.read_csv(to_modify_csv)

    # create date-time objects 
    to_modify_df['date'] = pd.to_datetime(to_modify_df['date'], format='%Y%m%d', errors='ignore')

    # sort by date and cik
    to_modify_df = to_modify_df.sort_values(by=['cik', 'date'], ascending=(True, True))

    # edit current csv file
    to_modify_df.to_csv(to_modify_csv)

def get_size(url):
    # try-except block to check URL validity:
    try:
        urlopen(url)
    except:
        return None

     # Open URL and store in file object
    file = urllib.request.urlopen(url)

    # fetching length in bytes
    return file.length

def get_size_scaled(to_read_csv='extracted_section_testS1s.csv', to_modify_csv='final_data_extended.csv'):
    to_read_df = pd.read_csv(to_read_csv)
    to_modify_df = pd.read_csv(to_modify_csv).drop_duplicates(subset=['cik']).set_index('cik')
    # sizes_df = DataFrame()
    sizes_df = pd.read_csv('s1_sizes.csv').drop_duplicates(subset=['cik']).set_index('cik')

    # sizes_df['cik'] = to_read_df['cik'].copy()
    # sizes_df['S-1 Size'] = to_read_df['localURL'].apply(get_size)
    # sizes_df = sizes_df.drop_duplicates(subset=['cik']).set_index('cik')
    # sizes_df.to_csv('s1_sizes.csv')
    # sizes_df = pd.read_csv('s1_sizes.csv')

    to_modify_df = to_modify_df.join(sizes_df, how='left').dropna(subset=['dateFiled'])
    # print(to_modify_df)
    to_modify_df.to_csv(to_modify_csv)
    
    print('Data for S-1 SIZES computed!')

def get_return(prices, time_interval):
    def convert_time_int(argument):
        switcher = {
            "weekly": 5,
            "monthly": 20,
            "semi-anually": 130,
            "yearly": 260
        }
        return switcher.get(argument, None)

    time_interval_int = convert_time_int(time_interval)
    t_0 = prices[0]
    try:
        t_interval = prices[time_interval_int - 1]
    except:
        return None
    return t_interval / t_0 - 1

def get_returns_scaled(time_interval, csv, to_modify_csv='returns.csv'):
    original_df = pd.read_csv(csv)
    to_modify_df = DataFrame()

    # get list of unique companies
    ciks = original_df['cik'].unique()
    
    # get_returns() for each company 
    rets = []
    for cik in ciks:
        prices = original_df[original_df['cik'] == cik]['prc'].to_list()
        # print(prices)
        ret = get_return(prices, time_interval)
        rets.append(ret)
    
    # print(rets)

    # create new dataframe for returns
    to_modify_df['cik'] = ciks
    to_modify_df['ret'] = rets

    # modify csv
    to_modify_df.to_csv(to_modify_csv)
    print('All RETURNS computed!')

# count_word_all()
# function: counts the number of occurrences of a word in the first 
# n lines of a URL
def count_word_all(url, words):
    # try-except block to check URL validity:
    try:
        urlopen(url)
    except:
        return None

     # Open URL and store in file object
    file = urllib.request.urlopen(url)
    data = str(file.read())

    # count occurrences 
    occurrences = []
    for word in words:
        occurrences.append(data.count(word))

    return ", ".join(occurrences)
    
def count_word_all_scaled(column_title, words, to_read_csv='extracted_section_testS1s.csv', to_modify_csv='final_data_extended.csv'):
    to_read_df = pd.read_csv(to_read_csv)
    to_modify_df = pd.read_csv(to_modify_csv)
    freq_df = DataFrame()

    freq_df['cik'] = to_read_df['cik'].copy()
    freq_df[column_title] = to_read_df['localURL'].apply(lambda x: count_word_all(x, words))
    freq_df = freq_df.drop_duplicates(subset=['cik']).set_index('cik')
    freq_df.to_csv('s1_freqs.csv')

    to_modify_df = to_modify_df.set_index('cik').join(freq_df, how='left').dropna(subset=['dateFiled'])
    print(to_modify_df)
    # to_modify_df.to_csv(to_modify_csv)
    print('Data for ' + ', '.join(words) + ' computed!')

def replace(dict_str):
    if type(dict_str) != str: return ""
    return str(dict_str).replace("'", '"')

def parse_emotion(dict_str, emotion):
    if type(dict_str) != str: return 0.0
    if not(dict_str): return 0.0

    emotions = json.loads(dict_str)
    return emotions[emotion]

def parse_emotion_scaled(emotion, csv='final_data_extended.csv'):
    df = pd.read_csv(csv)
    df[emotion] = df["RISK FACTORS Emotion(s)"].apply(replace).apply(lambda x: parse_emotion(x, emotion))
    df.to_csv(csv)
    print("Data for " + emotion.upper() + " extracted!")
    # print(df)

def plot_time_series(variable, csv='final_data_extended.csv'):
    df = pd.read_csv(csv)

    ############ TIME SERIES ############
    df['Year'] = df['dateFiled'].apply(lambda x : x[-4:])
    print(df)
    df1 = df.groupby("Year")[variable].mean().reset_index()
    # print(df1.Year)
    df1.plot.line(x='Year', y=(variable), subplots=False)
    plt.xlabel('Year')
    plt.ylabel(", ".join(variable))
    plt.show()
    #####################################

def plot_time_series_split(ave_var, variable, csv='final_data_extended.csv'):
    df = pd.read_csv(csv)
    df['Year'] = df['dateFiled'].apply(lambda x : x[-4:])
    ave = df[ave_var].mean()
    above_ave = df[df[ave_var] > ave]
    below_ave = df[df[ave_var] <= ave]

    ############ TIME SERIES ############
    df1 = above_ave.groupby("Year")[variable].mean().reset_index()
    df2 = below_ave.groupby("Year")[variable].mean().reset_index()
    # print(df1.Year)
    df1.plot.line(x='Year', y=(variable))
    plt.ylabel(", ".join(variable))
    df2.plot.line(x='Year', y=(variable))
    plt.xlabel('Year')
    plt.ylabel(", ".join(variable))
    plt.show()
    #####################################

def plot_relation(x_var, y_var, csv='final_data_extended.csv'):
    df = pd.read_csv(csv)

    df.plot.line(x=x_var, y=(y_var), subplots=False)
    plt.xlabel(x_var)
    plt.ylabel(", ".join(y_var))
    plt.show()

if __name__ == "__main__":
    to_modify_csv = 's1s_withPrices.csv'
    url = 'https://privatecapitaldatalab.com/private_edgar/edgar/data/1537069/000114420411070600/0001144204-11-070600'
    # sort_data_cik(to_modify_csv)
    # get_returns_scaled("semi-anually", to_modify_csv)
    # emotions = '{"Happy": 0.09, "Angry": 0.02, "Surprise": 0.13, "Sad": 0.11, "Fear": 0.65}'
    # parse_emotion_scaled("Fear")
    # csv = 'extracted_section_testS1s.csv'
    # get_size_scaled(csv)

    # plot_time_series(["RISK FACTORS Positive Sentiment", "RISK FACTORS Neutral Sentiment", "RISK FACTORS Negative Sentiment"])
    # plot_time_series(["1-Week Returns", "1-Month Returns", "6-Month Returns", "1-Year Returns"])
    # plot_time_series_split("Fear", ["1-Week Returns", "1-Month Returns", "6-Month Returns", "1-Year Returns"])
    x_var = 'Angry'
    y_var = '1-Year Returns'
    plot_relation('S-1 Size', ['1-Month Returns'], csv='final_data_extended.csv')
    # print(get_size(url))

    # get_size_scaled()
    # count_word_all_scaled("non-gaap, adjusted, unknown, uncertain, uncommon Frequencies", ["non-gaap", "adjusted", "unknown", "uncertain", "uncommon"])
    # count_word_all(url, ["non-gaap", "adjusted", "unknown", "uncertain", "uncommon"])


