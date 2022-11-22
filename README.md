# BlackRock-Caltech SURF Project

This is a compilation of all the code written during the 2021 Caltech Summer Undergraduate Research Fellowship. This is an independent research titled "Determining the Predictive Power of Pre-IPO Signals for Post-Public Stock Returns." As the name suggests, this project explores the ability to extract information and so-called 'signals' from pre-IPO data, such as S-1 or S-K filings, and correlate such extracted features with stock returns for multiple trading days. The abstract can reads:

> "The current method of trading sees many inefficiencies, specifically the lack of information about a company’s financial history, ownership structure, investment decisions, and previous investors. Such information may be crucial for investors towards decisions to purchasing securities. Therefore, this project investigates alternative forms of data, which can help bridge the gap between investor’s queries and knowledge about pre-IPO stock returns.

The first step is to gather varied sources such as registration statements (ie. S-1 filings with the Securities and Exchange Commission) published within a week before & after the IPO. The second step is to examine these financials and data from past IPOs, specifically how those companies performed before and after going public. The third step is to analyze which pre-IPO text variables from the S-1 filings hold predictive power. Such variables may include tone, sentiment, complexity, as well as frequency of certain keywords. Finally, an unsupervised machine learning algorithm will be developed to estimate post-public stock performance (specifically, stock price). So far, data has been collected on tone and sentiment (more to come), and analysis on the correlation between such variables and stock returns will be performed in the final upcoming weeks."

Many of the scripts written, such as ```calc_sent_edited_me.py```, ```word_search.py```, and ```data_analyzer.py``` primarily revolve my work trying to understand data analysis and build my own text data extractor. For the duration of the project, all of the data collected was done using ```s1_nlp_multithread_ME.py```. This includes all the packages I used, as well as the multithreaded techniques used to split the data files into batches and run them in parallel. 