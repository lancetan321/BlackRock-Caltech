import string
from typing import final
import itertools

import re
import emoji
import math

from collections import Counter
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from numpy import isnan
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus.reader.util import ConcatenatedCorpusView
from nltk.translate.phrase_based import extract

from pandas.core.frame import DataFrame
ps = PorterStemmer()
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from word_search import *
from bs4 import BeautifulSoup
import textstat
import collections
from collections import Counter
import text2emotion as te

# Watson Tone Analyzer Account Details
apikey = '7p0pf0BbFK36S1lSSssY3q5KSqXirtAXDaDXHaj1ov7l'
service_url = 'https://api.us-south.tone-analyzer.watson.cloud.ibm.com/instances/c69477b4-e9a0-4fd3-9566-51502036846f'

################# WEB SCRAPER #################

def remove_html(text_blob_str):
    # if NaN or not type 'str', return empty string
    if type(text_blob_str) != str: return ''

    # clean HTML tags out
    cleaned_tb_str = BeautifulSoup(text_blob_str, features='html5lib').get_text()

    return cleaned_tb_str

def remove_html_scaled(section_title, csv="extracted_section_testS1s.csv"):
    df = pd.read_csv(csv)
    df[section_title] = df[section_title].apply(remove_html)
    df.to_csv(csv)
    print(df)

def loc_ith_onwards(url, loc_ith):
    # try-except block to check URL validity:
    try:
        urlopen(url)
    except:
        return 0

    # Open URL and store in file object
    file = urllib.request.urlopen(url)
    time.sleep(1)

    # extract text from ith location onwards
    file_arr = file.readlines()
    file_size = len(file_arr)
    extracted_text = [file_arr[idx].decode('ISO-8859-1').strip('\n') for idx in range(loc_ith, file_size-1)]
    return extracted_text

def s1_extract(url, section_title, section_char_count, n=20000):
    # section-extracting algorithm:
    # 1) locate input title at start of text to extract
    # 2) record text to extract from
    # 3) extract text until section_char_count has been reached
    # 4) return text blob

    # Step 1: locate input title at start of text to extract
    if count_word(url, section_title, n) < 1:
        return ""
    elif count_word(url, section_title, n) == 1:
        loc_input_title = find_nth(url, section_title, 1)
    else:
        loc_input_title = find_nth(url, section_title, 2)

    # Step 2: record text to extract from
    rem_text = loc_ith_onwards(url, loc_input_title)

    # Step 3: extract text until section_char_count has been reached
    char_count = 0
    text_blob = []
    for line in rem_text:
        if char_count >= section_char_count:
            break
        text_blob.append(line)
        char_count += len(line)
    
    # Step 4: return text blob
    return " ".join(text_blob)

def scrape(section_title, section_char_count, sample_space=20, n=20000, csv='testS1s.csv'):
    # export csv to pandas dataframe
    df1 = pd.read_csv(csv)
    df1 = df1.sample(frac = 1)
    # df1 = df1[0:sample_space]
    
    # extract sections for all URLs and clean text of HTML syntax
    df2 = df1.copy()
    df2[section_title] = df2.localURL.apply(s1_extract, args = (section_title, section_char_count, n))

    # export to csv
    df1.to_csv('basic_nlp_results.csv')
    df2.to_csv("extracted_section_testS1s.csv")
    print('Scraping completed!')

########### FILE LOCATION APPENDER ###########

def file_loc_append(file_loc_txt):
    return "/public_html/Archives/" + file_loc_txt

def file_loc_append_scaled(csv='extracted_section_testS1s.csv'):
    to_modify_df = pd.read_csv(csv)
    to_modify_df['fileLocation'] = to_modify_df['fileLocation'].apply(file_loc_append)
    to_modify_df.to_csv(csv)
    print('FILE LOCATION APPENDED!')

################# SENTIMENT ANALYZER #################

analyser = SentimentIntensityAnalyzer()

# Source (NLTK): https://gaurav5430.medium.com/using-nltk-for-lemmatizing-sentences-c1bfff963258

def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:          
        return None

def lemmatize_text(sentence):
    #instantiate lemmatizer
    lem = WordNetLemmatizer()

    #tokenize the sentence and find the POS tag for each token
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))  
    #tuple of (token, wordnet_tag)
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        # print("word: ", word)
        # print("tag: ", tag)
        if tag is None:
            # if there is no available tag, append the token as is
            lemmatized_sentence.append(word)
        else:        
            # else use the tag to lemmatize the token
            lemmatized_sentence.append(lem.lemmatize(word, tag))
    return " ".join(lemmatized_sentence)

# lexicon-based sentiment analyzer
def sentiment_analyse(text, positive='positive.txt', negative='negative.txt'):
    # text cleaning
    cleaned_text = list(map(lambda x : x.translate(str.maketrans('', '', string.punctuation)), nltk.sent_tokenize(text)))
    # print(cleaned_text)
    
    # lemmatization (including POS tagging)
    lemmatized_text = list(map(lambda x : lemmatize_text(x), cleaned_text))
    
    # print(lemmatized_text)

    # remove stopwords
    stop_words = set(stopwords.words('english'))
    lemmatized_text = list(map(lambda x : nltk.word_tokenize(x), lemmatized_text))
    lemmatized_text = list(itertools.chain.from_iterable(lemmatized_text))
    lemmatized_text = list(map(lambda x : x.lower(), lemmatized_text))
    filtered_text = [w for w in lemmatized_text if w not in stop_words]

    # print(filtered_text)

    # count how many positive and negative words there are
    positive_words = open(positive, encoding='utf-8').read().split('\n')
    negative_words = open(negative, encoding='utf-8').read().split('\n')
    positives = []
    negatives = []
    for w in filtered_text:
        if w in negative_words:
            negatives.append(w)
        if w in positive_words:
            positives.append(w)

    # calculate sentiment score
    sentiment_score = len(positives)/(len(negatives) + 1)
    return sentiment_score

# VADER sentiment analyzers
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
 
############# SUBJECTIVITY ANALYZER #############

def analyze_subjectivity(html_free_text):
    # if NaN or not type 'str', return 0.0
    if type(html_free_text) != str: return 0.0

    return TextBlob(html_free_text).sentiment.subjectivity

################ EMOTION ANALYZER ################

def analyze_emotion(html_free_text, stop):
    # if NaN or not type 'str', return ''
    if type(html_free_text) != str or html_free_text == 'nan': return ''

    return te.get_emotion(html_free_text, stop)

############### COMPLEXITY ANALYZER ###############

def calc_complexity(cleaned_tb_str):
    # if NaN or not type 'str', return 0.0
    if type(cleaned_tb_str) != str: return 999.0

    return textstat.flesch_reading_ease(cleaned_tb_str)

############# BAG OF WORDS ANALYZER #############

def top_five_words(html_free_text, useless='useless_words.txt'):
    # if NaN or not type 'str', return 0.0
    if type(html_free_text) != str: return ''

    # text cleaning
    cleaned_text = list(map(lambda x : x.translate(str.maketrans('', '', string.punctuation)), nltk.sent_tokenize(html_free_text)))
    # print(cleaned_text)
    
    # lemmatization (including POS tagging)
    lemmatized_text = list(map(lambda x : lemmatize_text(x), cleaned_text))
    
    # print(lemmatized_text)

    # remove stopwords
    stop_words = set(stopwords.words('english'))
    lemmatized_text = list(map(lambda x : nltk.word_tokenize(x), lemmatized_text))
    lemmatized_text = list(itertools.chain.from_iterable(lemmatized_text))
    lemmatized_text = list(map(lambda x : x.lower(), lemmatized_text))
    filtered_text = [w for w in lemmatized_text if w not in stop_words]

    # remove useless words
    Counter = collections.Counter(filtered_text)
    useless_words = open(useless, encoding='utf-8').read().split('\n')
    most_freq = Counter.most_common(200)
    five_most_freq = []
    for word_tuple in most_freq:
        if len(five_most_freq) >= 5: break
        word = word_tuple[0]
        if (word not in useless_words) and (not word.isdigit()) and (any(c.isalnum() for c in word)): 
            five_most_freq.append(word_tuple)

    return str(five_most_freq).replace('[','').replace(']','')

########### STANDARD INDUSTRIAL CLASSIFICATION ###########

def extract_sic(url):
    # extract the lines
    lines = loc_ith_onwards(url, 0)

    # extract standard industry classification
    for line in lines:
        if "standard industrial classification" in line.lower() and len(line.lower()) <= 100:
            sic_num = ''
            for charAtI in line.lower():
                if charAtI.isdigit(): sic_num += charAtI
            return sic_num

    return ''

################################ THREADED FUNCTIONS ################################

def extract_sic_scaled_threaded(section_title, file_name='in0.csv', to_modify='out0.csv'): 
    sections_df = pd.read_csv(file_name)
    to_modify_df = DataFrame()

    to_modify_df["SIC Number"] = sections_df.localURL.apply(extract_sic).tolist()
    to_modify_df.to_csv(to_modify)
    print("Data for SIC NUMBERs extracted from " + file_name)

def calc_sent_scaled_threaded(section_title, file_name='in0.csv', to_modify='out0.csv'):
    sections_df = pd.read_csv(file_name)
    to_modify_df = DataFrame()
    extracted_sentiments = DataFrame()

    extracted_sentiments["sentiments"] = sections_df[section_title].apply(calc_sentiment)
    to_modify_df[section_title + " Positive Sentiment"] = extracted_sentiments["sentiments"].apply(parse_data_pos).tolist()
    to_modify_df.to_csv(to_modify)
    print("Data for POSITIVE SENTIMENTS extracted from " + file_name)
    to_modify_df[section_title + " Neutral Sentiment"] = extracted_sentiments["sentiments"].apply(parse_data_neu).tolist()
    to_modify_df.to_csv(to_modify)
    print("Data for NEUTRAL SENTIMENTS extracted from " + file_name)
    to_modify_df[section_title + " Negative Sentiment"] = extracted_sentiments["sentiments"].apply(parse_data_neg).tolist()
    to_modify_df.to_csv(to_modify)
    print("Data for NEGATIVE SENTIMENTS extracted from " + file_name)
    print("Data for all 3 SENTIMENTS extracted from " + file_name)

def analyze_emotion_scaled_threaded(section_title, file_name='in0.csv', to_modify='out0.csv'):
    sections_df = pd.read_csv(file_name)
    to_modify_df = DataFrame()

    stop_words = set(stopwords.words('english'))
    stop = [x.lower() for x in stop_words]
    to_modify_df[section_title + " Emotion(s)"] = sections_df[section_title].apply(lambda x: analyze_emotion(x, stop)).tolist()
    to_modify_df.to_csv(to_modify)
    print("Data for EMOTIONS extracted from " + file_name)

def calc_complexity_scaled_threaded(section_title, file_name='in0.csv', to_modify='out0.csv'):
    sections_df = pd.read_csv(file_name)
    to_modify_df = DataFrame()
    to_modify_df[section_title + " Complexity"] = sections_df[section_title].apply(calc_complexity).tolist()
    to_modify_df.to_csv(to_modify)
    print("Data for COMPLEXITIES from " + file_name)

def analyze_subjectivity_scaled_threaded(section_title, file_name='in0.csv', to_modify='out0.csv'):
    sections_df = pd.read_csv(file_name)
    to_modify_df = DataFrame()

    to_modify_df[section_title + " Subjectivity"] = sections_df[section_title].apply(analyze_subjectivity).tolist()
    to_modify_df.to_csv(to_modify)
    print("Data for SUBJECTIVITIES from " + file_name)

def top_five_words_scaled_threaded(section_title, file_name='in0.csv', to_modify='out0.csv'):
    sections_df = pd.read_csv(file_name)
    to_modify_df = DataFrame()

    to_modify_df[section_title + " Top 5 Words"] = sections_df[section_title].apply(top_five_words).tolist()
    to_modify_df.to_csv(to_modify)
    print("Data for BAG-OF-WORDS extracted from " + file_name)

import threading 
from datetime import datetime

def start_threaded_task(section_title, func, num_files=1, in_file_prefix="in", out_file_prefix="out"):
    for i in range(num_files):
        wordnet.ensure_loaded()
        a = threading.Thread(target=func, args=(section_title, in_file_prefix + str(i)+ ".csv", out_file_prefix + str(i)+ ".csv", ))
        a.start()
        print("Started thread " +str(i) + " at " +  datetime.now().strftime("%H:%M:%S"))

################## COMMAND LINE ##################

if __name__ == "__main__":
    url = "https://privatecapitaldatalab.com/private_edgar/edgar/data/1355839/000121390013005509/0001213900-13-005509.txt"
    # scrape("RISK FACTORS", 164855, 40)
    # scrape("RISK FACTORS", 164855)
    # remove_html_scaled('RISK FACTORS')
    # file_loc_append_scaled()

    # extract_sic_scaled()
    # calc_sent_scaled('RISK FACTORS')
    # analyze_emotion_scaled('RISK FACTORS')
    # calc_complexity_scaled('RISK FACTORS')
    # top_five_words_scaled('RISK FACTORS')

    # start_threaded_task("RISK FACTORS", analyze_subjectivity_scaled_threaded, num_files=100)

    bigger = r"C:\Users\lance\Desktop\SURF_2021\Output\complexity_emotion_sic.csv"
    smaller = "sent_subject.csv"
    returns = 'returns.csv'
    final_data_csv = "final_data.csv"
    to_modify_csv = "final_data_extended.csv"
    # to_modify_csv=r"C:\Users\lance\Desktop\SURF_2021\Output\complexity_emotion_sic.csv"
    # bigger_df = pd.read_csv(bigger).drop_duplicates(subset=['cik']).set_index('cik')
    # smaller_df = pd.read_csv(smaller).drop_duplicates(subset=['cik']).set_index('cik')

    # returns_df = pd.read_csv(returns).drop_duplicates(subset=['cik']).set_index('cik')
    to_modify_df_og = pd.read_csv(to_modify_csv).drop_duplicates(subset=['cik']).set_index('cik')
    final_data_df = pd.read_csv(final_data_csv)
    variable_df = final_data_df[['cik', 'Photos Frequencies', "non-gaap Frequency", "adjusted Frequency", "unknown Frequency", "uncertain Frequency", "uncommon Frequency"]].copy().set_index('cik')

    # to_modify_df = bigger_df.join(smaller_df, how='left').join(returns_df, how='left')
    # to_modify_df = to_modify_df_og.join(returns_df, how='left')
    to_modify_df = to_modify_df_og.join(variable_df, how='left')
    # print(to_modify_df)
    to_modify_df.to_csv(to_modify_csv)

    # print(returns_df)
    # to_modify_df_og = pd.read_csv(to_modify_csv).set_index('cik')
    # print(to_modify_df_og)
    # to_modify_df = to_modify_df_og.join(returns_df, how='right').join(returns_df, how='left').dropna(subset=['dateFiled'])
    # to_modify_df = to_modify_df_og.join(returns_df, how='left').dropna(subset=['dateFiled'])
    # to_modify_df.to_csv(to_modify_csv)
    # to_modify_df_og = pd.read_csv(to_modify_csv)

    # to_modify_df.to_csv(to_modify_csv)
    # print(to_modify_df)
    # print(all(elem in bigger_list  for elem in smaller_list))
    # print(bigger_df)
    # print(smaller_df)
    # df1 = pd.read_csv(to_modify_csv)
    print('DONE!')