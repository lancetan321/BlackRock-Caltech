import pandas as pd
from pandas.core.frame import DataFrame
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import threading 
from datetime import datetime

analyser = SentimentIntensityAnalyzer()

def calc_sent_edited_me(section_title, file_name='in0.csv', to_modify='out0.csv'):
    main = pd.read_csv(file_name)
    #print(main.head())

    sections_df = main[['localURL', section_title]]
    to_modify_df = DataFrame()

    length = len(sections_df)
    for i in range(0,length):
        print(f"Working on sentiment {i}")
        cleaned_tb_str = sections_df[section_title][i]
        if type(cleaned_tb_str) != str:
            to_save = "0.0|0.0|0.0"
        else:
            sentiment_dict = analyser.polarity_scores(cleaned_tb_str)
            sentiments = [sentiment_dict['pos'], sentiment_dict['neu'], sentiment_dict['neg']]
            to_save = "|".join([str(x) for x in sentiments])

        to_modify_df["localURL"] = sections_df["localURL"]
        to_modify_df["sentiments"] = to_save

    to_modify_df.to_csv(to_modify)
    print('Data for SENTIMENT extracted for ' + file_name + '!')

def start_threaded_task(section_title, func, num_files=1, in_file_prefix="in", out_file_prefix="out"):
    for i in range(num_files):
        a = threading.Thread(target=func, args=(section_title, in_file_prefix + str(i)+ ".csv", out_file_prefix + str(i)+ ".csv", ))
        a.start()
        print("Started thread " +str(i) + " at " +  datetime.now().strftime("%H:%M:%S"))

if __name__ == "__main__":
    start_threaded_task('RISK FACTORS', calc_sent_edited_me, 100)
    # print(pd.read_csv('out66.csv'))
#    calc_sent_edited(csv)
