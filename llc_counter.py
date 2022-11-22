import pandas as pd
# import nltk 
# from nltk.stem import WordNetLemmatizer
# from textblob import TextBlob, Word

# txt = open('positive.txt', encoding='utf-8').read().split('\n')

# # x = txt.split()

# word = 'jumped'
# w = Word(word)
# print(w.lemmatize())


# risk_factors_1 = ""


movie_titles_df = pd.read_csv('movie_titles.csv', header=None)
print(movie_titles_df)