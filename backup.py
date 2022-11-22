   # initialize list of word counts
    first_occ = []
    
    # list of nth occurrence of word
    nth_occurrences = [] 

    # list of number of occurrences of word in the last k lines of the file
    last_occ = []
   
   # iterate through localURLs and count_words()
    for i, url in enumerate(df["localURL"]):
        if i % 10 == 0: 
            print(str(i) + " out of 21363")

        time.sleep(1)
        first_occ.append(count_word(url, w, n))
        nth_occurrences.append(find_nth(url, w, nth))
        last_occ.append(count_word_last(url, w, k))
   
   
    # add list as new column to df
    df[w + " " + "NumOccurrencesInFirstNLines"] = first_occ
    df[w + " " + "NthOccurrenceLineNumber"] = nth_occurrences
    df[w + " " + "NumOccurrencesInLastNLines"] = last_occ


######## WEB SCRAPER ########

for ith in range(0, count_word(url, section_title, n)):
        loc_input_title = find_nth(url, section_title, ith)                         # location of the ith instance
        prev_text = record_start_end(url, loc_input_title - 20, loc_input_title)    # record all lines up to loc_ith in array
        if is_toc_pb_cont(prev_text):                                                          # scan previous 20 lines and check if either page-break or table of contents is contained
            break



#############################

# result = requests.get(url)

# print(result.status_code)

# print(result.headers)

# src = result.content
# soup = BeautifulSoup(src, 'lxml')

# stuff = soup.find_all("h5")
# print(stuff)
# print("\n")

# section_titles = []
# for center_tag in soup.find_all("center"):
#     title = center_tag.find("b")
#     section_titles.append(title.attrs["b"])