# # from word_search import word_search
# import seaborn as sns
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import os
# import urllib
# import urllib.request
# from urllib.request import urlopen
# from word_search import *
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# import nltk
# from bs4 import BeautifulSoup

# url = "https://privatecapitaldatalab.com/private_edgar/edgar/data/1537069/000114420411070600/0001144204-11-070600.txt"



# def concate_test(w):
#     for i, word in enumerate(w):
#         w[i] += 'hi'

#     return w

# def ranger():
#     sample_arr = [1, 2, 3]
#     a = False
#     # print(range(len(sample_arr) - 0, len(sample_arr) - 1))
#     for x in range(len(sample_arr) - 1, len(sample_arr) - 2, -1): 
#         if sample_arr[x] > 2:
#             a = True
    
    

#     return a

# def count_chars(txt):
#     file = open(txt)
#     data = file.read()
#     num_of_char = len(data)

#     return num_of_char

# def print_html(url):
#     file = urllib.request.urlopen(url)
#     time.sleep(1)
#     string = ""
#     for line in file:
#         # decode the line and count word occurrences
#         string += line.decode("utf-8").strip('\n')
#         print(string)
#         time.sleep(1)

# def shuffle_rows(csv):
#     df = pd.read_csv(csv)
#     df = df.sample(frac = 1)
#     df = df[0:25]
#     df.to_csv("risks_to_extract.csv")
#     print(df)

# ############3 CLEAN HTML TEXT #############

# def clean_html(csv):
#     df = pd.read_csv(csv)
#     # print(df['Risk Factors'])
    
#     df['Filtered'] = df['RISK FACTORS'].apply(lambda x: BeautifulSoup(x, features='html5lib').get_text())

#     # soup = BeautifulSoup(str, features='html5lib').get_text()
#     print(df)

# ######## VADER SENTIMENT ANALYZER ########

# def calc_sentiment(sentence):
#     analyser = SentimentIntensityAnalyzer()
#     sentiment_dict = analyser.polarity_scores(sentence)
#     print("Overall sentiment dictionary is : ", sentiment_dict)
#     print("sentence was rated as ", sentiment_dict['neg']*100, "% Negative")
#     print("sentence was rated as ", sentiment_dict['neu']*100, "% Neutral")
#     print("sentence was rated as ", sentiment_dict['pos']*100, "% Positive")
 
#     print("Sentence Overall Rated As", end = " ")
 
#     # decide sentiment as positive, negative and neutral
#     if sentiment_dict['compound'] >= 0.05 :
#         print("Positive")
 
#     elif sentiment_dict['compound'] <= - 0.05 :
#         print("Negative")
 
#     else :
#         print("Neutral")

# ##########################################

# def extract_cik_fd(to_extract_csv='extracted_section_testS1s.csv'):
#     to_extract_df = pd.read_csv(to_extract_csv)
#     extracted_cik_fd = to_extract_df[["cik", "dateFiled"]]
#     extracted_cik_fd.to_csv('extracted_cik_fd.csv')

# extract_cik_fd()

# # risk factor section lengths: 166410, 140514, 41233, 134177, 60387, 215655, 121796, 438669

# # print(ranger())
# # ranger()
# # print(count_chars('risk_factors_1.txt'))
# # print(count_word(url, "RISK FACTORS", 30000))
# # shuffle_rows("s1s.csv")
# # a = sum([166410, 140514, 41233, 134177, 60387, 215655, 121796, 438669])
# # b = len([166410, 140514, 41233, 134177, 60387, 215655, 121796, 438669])
# # ave = a / b
# sample_str = "<TITLE>VintageFilings,LLC</TITLE>"
# html = ""
# # print_html(url)
# sent = "I am a good person!"
# # calc_sentiment(sent)
# # clean_html('basic_nlp_results.csv')
# # clean_html(sample_str)

# # url = "https://privatecapitaldatalab.com/private_edgar/edgar/data/1538123/000153812311000002/0001538123-11-000002.txt"
# # df = pd.read_csv('in2.csv')
# # print(df)

# print(pd.to_datetime(['20050527', '20060527'], format='%Y%m%d', errors='ignore'))

import nltk

lines = 'university of pennsylvania students are the cream of the crop'
found = lambda pos: pos[:2] == 'NN'
tokenized = nltk.word_tokenize(lines)
tagged = nltk.pos_tag(tokenized)

z = [word for (word, pos) in tagged if found(pos)]
l = len (tagged)
print(z,'\n', l)