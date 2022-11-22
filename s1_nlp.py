import string
from typing import final
import itertools
from nltk.corpus.reader.util import ConcatenatedCorpusView
from nltk.translate.phrase_based import extract
from collections import Counter
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
ps = PorterStemmer()
from nltk.stem.wordnet import WordNetLemmatizer
lem = WordNetLemmatizer()
from nltk.corpus import wordnet
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from word_search import *
from bs4 import BeautifulSoup
from ibm_watson import ToneAnalyzerV3
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
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
    #tokenize the sentence and find the POS tag for each token
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))  
    #tuple of (token, wordnet_tag)
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        # print("word: ", word)
        # print("tag: ", tag)
        if tag is None:
            #if there is no available tag, append the token as is
            lemmatized_sentence.append(word)
        else:        
            #else use the tag to lemmatize the token
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
def calc_sentiment_pos(cleaned_tb_str):
    # if NaN or not type 'str', return 0.0
    if type(cleaned_tb_str) != str: return 0.0

    # initiate analysis
    sentiment_dict = analyser.polarity_scores(cleaned_tb_str)
    return sentiment_dict['pos']
def calc_sentiment_neu(cleaned_tb_str):
    # if NaN or not type 'str', return 0.0
    if type(cleaned_tb_str) != str: return 0.0

    # initiate analysis
    sentiment_dict = analyser.polarity_scores(cleaned_tb_str)
    return sentiment_dict['neu']
def calc_sentiment_neg(cleaned_tb_str):
    # if NaN or not type 'str', return 0.0
    if type(cleaned_tb_str) != str: return 0.0

    # initiate analysis
    sentiment_dict = analyser.polarity_scores(cleaned_tb_str)
    return sentiment_dict['neg']

def calc_sent_scaled(section_title, sections='extracted_section_testS1s.csv', to_modify='basic_nlp_results.csv'):
    sections_df = pd.read_csv(sections)
    to_modify_df = pd.read_csv(to_modify)
    to_modify_df[section_title + " Positive Sentiment"] = sections_df[section_title].apply(calc_sentiment_pos).tolist()
    to_modify_df[section_title + " Neutral Sentiment"] = sections_df[section_title].apply(calc_sentiment_neu).tolist()
    to_modify_df[section_title + " Negative Sentiment"] = sections_df[section_title].apply(calc_sentiment_neg).tolist()
    to_modify_df.to_csv('basic_nlp_results.csv')
    print('Data for all 3 SENTIMENTS extracted!')
 
################ EMOTION ANALYZER ################

def analyze_emotion(html_free_text):
    # if NaN or not type 'str', return 0.0
    if type(html_free_text) != str: return ''

    return te.get_emotion(html_free_text)

def analyze_emotion_scaled(section_title, sections='extracted_section_testS1s.csv', to_modify='basic_nlp_results.csv'):
    sections_df = pd.read_csv(sections)
    to_modify_df = pd.read_csv(to_modify)

    to_modify_df[section_title + " Emotion(s)"] = sections_df[section_title].apply(analyze_emotion).tolist()
    to_modify_df.to_csv('basic_nlp_results.csv')
    print('Data for EMOTIONS extracted!')

################## TONE ANALYZER ##################

# IBM Watson Tone Analyzer 
def analyze_tone(text, n=130000):
    # if NaN or not type 'str', return empty character
    if type(text) != str: return ''

    # authenticate
    authenticator = IAMAuthenticator(apikey)
    ta = ToneAnalyzerV3(version='2017-09-21', authenticator=authenticator)
    ta.set_service_url(service_url)

    # put sentences in list
    tok_list = nltk.sent_tokenize(text)
    char_count = 0
    sent_list = []
    for sentence in tok_list:
        if char_count >= n:
            break
        sent_list.append(sentence)
        char_count += len(sentence)

    # analyze tone
    document_tones = list(map(lambda x: ta.tone(x).get_result(), sent_list))
    tone_names = [res['document_tone']['tones'][0]['tone_name'] for res in document_tones if len(res['document_tone']['tones']) > 0]

    # res = ta.tone(x).get_result()
    # tones = res['document_tone']['tones']
    # tone_names = []
    # for tone in tones:
    #     tone_names.append(tone['tone_name'])

    return ', '.join(list(set(tone_names)))

def analyze_tone_scaled(section_title, sections='extracted_section_testS1s.csv', to_modify='basic_nlp_results.csv'):
    sections_df = pd.read_csv(sections)
    to_modify_df = pd.read_csv(to_modify)
    print(sections_df)

    to_modify_df[section_title + " Tone(s)"] = sections_df[section_title].apply(analyze_tone).tolist()
    to_modify_df.to_csv('basic_nlp_results.csv')
    print('Data for TONES extracted!')

############### COMPLEXITY ANALYZER ###############

def calc_complexity(cleaned_tb_str):
    # if NaN or not type 'str', return 0.0
    if type(cleaned_tb_str) != str: return 999.0

    return textstat.flesch_reading_ease(cleaned_tb_str)

def calc_complexity_scaled(section_title, sections='extracted_section_testS1s.csv', to_modify='basic_nlp_results.csv'):
    sections_df = pd.read_csv(sections)
    to_modify_df = pd.read_csv(to_modify)
    to_modify_df[section_title + " Complexity"] = sections_df[section_title].apply(calc_complexity).tolist()
    to_modify_df.to_csv('basic_nlp_results.csv')
    print('Data for COMPLEXITIES extracted!')

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

def top_five_words_scaled(section_title, sections='extracted_section_testS1s.csv', to_modify='basic_nlp_results.csv'):
    sections_df = pd.read_csv(sections)
    to_modify_df = pd.read_csv(to_modify)

    to_modify_df[section_title + " Top 5 Words"] = sections_df[section_title].apply(top_five_words).tolist()
    to_modify_df.to_csv('basic_nlp_results.csv')
    print('Data for BAG-OF-WORDS extracted!')

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

def extract_sic_scaled(sections='extracted_section_testS1s.csv', to_modify='basic_nlp_results.csv'):
    sections_df = pd.read_csv(sections)
    to_modify_df = pd.read_csv(to_modify)

    to_modify_df["SIC Number"] = sections_df.localURL.apply(extract_sic).tolist()
    print(to_modify_df)
    to_modify_df.to_csv('basic_nlp_results.csv')
    print('Data for SIC NUMBERs extracted!')

################## COMMAND LINE ##################

if __name__ == "__main__":
    url = "https://privatecapitaldatalab.com/private_edgar/edgar/data/1355839/000121390013005509/0001213900-13-005509.txt"
    # scrape("RISK FACTORS", 164855, 40)
    scrape("RISK FACTORS", 164855)
    remove_html_scaled('RISK FACTORS')
    file_loc_append_scaled()

    # df = pd.read_csv('extracted_section.csv')
    # df2 = pd.read_csv('basic_nlp_results.csv')
    # print(df2)

    # extract_sic_scaled()
    # calc_sent_scaled('RISK FACTORS')
    # analyze_emotion_scaled('RISK FACTORS')
    # calc_complexity_scaled('RISK FACTORS')
    # top_five_words_scaled('RISK FACTORS')

    # average length of risk factors =  164855
    # text = "Hello, Lance. So sad to see you go. Can we dance tonight????"
    # print(top_five_words(text))
    # analyze_tone(text)
    # print(textstat.automated_readability_index(text))

    # print(BeautifulSoup(s1_extract(url, 'RISK FACTORS', 164855), 'html5lib').get_text())