import pandas as pd
from pandas.core.frame import DataFrame
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyser = SentimentIntensityAnalyzer()

def calc_sentiment(cleaned_tb_str):
    # if NaN or not type 'str', return 0.0,0.0,0.0
    if type(cleaned_tb_str) != str: return "0.0,0.0,0.0"

    # initiate analysis
    sentiment_dict = analyser.polarity_scores(cleaned_tb_str)
    sentiments = [sentiment_dict['pos'], sentiment_dict['neu'], sentiment_dict['neg']]
    return ",".join([str(x) for x in sentiments])

def parse_data_pos(sentiment_data):
    sentiment_list = str(sentiment_data).split(",")
    parsed = list(map(float, sentiment_list))
    return parsed[0]
def parse_data_neu(sentiment_data):
    sentiment_list = str(sentiment_data).split(",")
    parsed = list(map(float, sentiment_list))
    return parsed[1]
def parse_data_neg(sentiment_data):
    sentiment_list = str(sentiment_data).split(",")
    parsed = list(map(float, sentiment_list))
    return parsed[2]


def calc_sent_edited(section_title, file_name='ext', to_modify='out0.csv'):
    sections_df = pd.read_csv(file_name)
    to_modify_df = DataFrame()
    extracted_sentiments = DataFrame()

    extracted_sentiments["sentiments"] = sections_df[section_title].apply(calc_sentiment)
    to_modify_df[section_title + " Positive Sentiment"] = extracted_sentiments["sentiments"].apply(parse_data_pos).tolist()
    to_modify_df[section_title + " Neutral Sentiment"] = extracted_sentiments["sentiments"].apply(parse_data_neu).tolist()
    to_modify_df[section_title + " Negative Sentiment"] = extracted_sentiments["sentiments"].apply(parse_data_neg).tolist()
    to_modify_df.to_csv(to_modify)
    print("Data for all 3 SENTIMENTS extracted from " + file_name)

if __name__ == "__main__":
    csv = ''
    # calc_sent_edited(csv)