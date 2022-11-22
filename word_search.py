# read
import urllib
import urllib.request
from urllib.request import urlopen
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime as dt

# count_words()
# function: counts the number of occurrences of a word in the first 
# n lines of a URL
def count_word(url, w, n):
    # try-except block to check URL validity:
    try:
        urlopen(url)
    except:
        return -2

     # Open URL and store in file object
    file = urllib.request.urlopen(url)
    time.sleep(1)
    i = n 
    count = 0
    for line in file:
        # quit if past ith line
        if i <= 0:
            break

        # decode the line and count word occurrences
        decoded_line = line.decode("utf-8")
        count += decoded_line.count(w)

        i -= 1

    return count

# count_word_last()
# function: returns the count of the query string w in the last N lines(1-indexed) of the document.
# Default: N = number of lines in the document, so the whole document is searched 
# returns 0 for no match
def count_word_last(url, w, n):
    # try-except block to check URL validity:
    try:
        urlopen(url)
    except:
        return -2

    file = urllib.request.urlopen(url)
    time.sleep(1)
    num_total_lines = len(urllib.request.urlopen(url).readlines())
    line_num, count = 0, 0
    for line in file:
        line_num += 1
        if line_num <num_total_lines - n:
            continue
        count += line.decode("utf-8").count(w)
    return count
    
# find_nth()
# function: returns the line number corresponding to the nth occurrence
# of the query string w, in the specified url. 
# Default: n = 1 (line number corresponding to the first occurrence of the word if not specified)
# returns -1 if the word was not found in any line
def find_nth(url, w, n=1):
    # try-except block to check URL validity:
    try:
        urlopen(url)
    except:
        return -2

    file = urllib.request.urlopen(url)
    time.sleep(1)
    l_num, occ_num = 0, 0
    
    for line in file:
        l_num += 1
        if w not in line.decode("utf-8"):
            continue
        occ_num +=  line.decode("utf-8").count(w)
        if occ_num >= n:
            return l_num
    return -1

# word_search()
# function: 
# - counts the number of occurrences of a word in the first
#   n lines of a URL for every URL in localURL
# - counts the number of occurrences of a word in the last
#   n lines of a URL for every URL in localURL
# - searches for nth instance --> returns the corresponding line #
# - 
def word_search(csv, w, n, nth, k): 
    # convert csv to pandas df
    df = pd.read_csv(csv)
    df = df.sample(frac = 1)
    df = df[0:999]

    for word in w:
        # apply() previously defined functions to analyze S-1 data
        df[word + " " + "NumOccurrencesInFirstNLines"] = df.localURL.apply(count_word, args = (word, n))
        df[word + " " + "NthOccurrenceLineNumber"] = df.localURL.apply(find_nth, args = (word, nth))
        df[word + " " + "NumOccurrencesInLastNLines"] = df.localURL.apply(count_word_last, args = (word, k))

        ############ TIME SERIES ############
        df['Year'] = df['dateFiled'].apply(lambda x : x[-4:])
        df1 = df.groupby("Year")[word + " " + "NumOccurrencesInFirstNLines"].sum().reset_index()
        df1 = df1.set_index("Year")
        df1.plot(kind = 'bar', figsize=(5, 5))
        plt.xlabel('Year')
        plt.ylabel(word + " count")
        plt.show()
        #####################################
        print(df1)

    # export df to new csv

    # rename col A
    df.to_csv("results.csv")
    print(df)

if __name__ == "__main__":
    url = "https://privatecapitaldatalab.com/private_edgar/edgar/data/1537069/000114420411070600/0001144204-11-070600.txt"
    # word_search("s1s.csv", ["LLC"], 1000, 0, 0)
    # print(count_word(url, "RISK FACTORS", 20000))