#!/usr/bin/env python
# coding: utf-8

# Nowadays companies want to understand, what went wrong with their latest products? What users and the general public think about the latest feature? You can quantify such information with reasonable accuracy using sentiment analysis.
# 
# Quantifying users content, idea, belief, and opinion is known as sentiment analysis. User's online post, blogs, tweets, feedback of product helps business people to the target audience and innovate in products and services. Sentiment analysis helps in understanding people in a better and more accurate way. It is not only limited to marketing, but it can also be utilized in politics, research, and security.
# 
# There are mainly two approaches for performing sentiment analysis.
# 
# Lexicon-based: count number of positive and negative words in given text and the larger count will be the sentiment of text.
# 
# Machine learning based approach: Develop a classification model, which is trained using the pre-labeled dataset of positive, negative, and neutral.
# 
# In this execrsice, you will use the first approach(Machine learning based approach). This is how you learn sentiment and text classification with a single example.

# Download all the packages from NLTK.

# In[1]:


import nltk
#nltk.download('punkt')
import pandas as pd

# Pre-Processing using NLTK 
# We have extracted review comment from amazon for on the product. This is Digital Thermostat Control Unit - A419ABG-3C. We have written these review comment in and CSV file named 'AmazonComments.csv'. We can now start preprocess these reviews for text analytics and sentiment anlaysis.

# In[2]:


#read reviews from csv
path='FILEPATH\\AmazonComments.csv'
review = pd.read_csv(path,encoding="ISO-8859-1")
review.columns=['Review Number','Review']
review.head()


# In[3]:


#remove observation column
review.drop(["Review Number"], axis = 1, inplace = True) 
review.head()


# Import NLTK packages to perform analysis of the review dataframe. i.e review['Review']

# In[16]:


#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk import tokenize


# In[21]:


#join all the review in one big text which can be tokenized by sentences later.
AllReview_text = " ".join(review for review in review['Review'])
print(AllReview_text)


# Since the word tokenization also tokenize punctuations. May want to remove those first, maybe also remove numbers. Below will remove everything other than text from the each row. replace('[^A-z ]',' ') replaces anything string other than alphabet with space.

# In[6]:


#using regular expression removing punctuation and numbers from each of the review comments.
review['Review_only_text']=review['Review'].str.replace('[^A-z ]',' ').str.replace(' +',' ').str.strip()


# In[7]:


#view 1st comments after cleaining text
review['Review_only_text'][1]


# Lets create one string with all the review comments from data frame review['Review']. This text can then be processed to remove stopwords and create word cloud.

# In[9]:


text = " ".join(review for review in review['Review_only_text'])
type(text)
#str
text.lower()
text


# Now, use Word tokenizer to get list of each words seperated by spaces.

# In[18]:


#review['tokenized_sents'] = review['Review'].apply(lambda row: word_tokenize(row))
tokenized_word=tokenize.word_tokenize(text)
print(tokenized_word)


# In[22]:


#use AllReview_text to tokenize by sentences for sentiment analysis using VADER.
tokenized_sent=tokenize.sent_tokenize(AllReview_text)
print(tokenized_sent)


# In[82]:


type(tokenized_word)


# Let's get rid of stop words.

# In[76]:


stop_words = set(stopwords.words('english')) 
# from the above tokens we lot of references of words like "I","Temperature","temp","Controller","thermostat". 
#Since these reviews are realted to thermostat we can remove these word to analyse actual experience of customer.

stop_words.update(["I","temperature","temp","controller","thermostat","The"])

#tokenized_word_without_Stopwords = [w for w in tokenized_word if not w in stop_words] 
  
tokenized_word_without_Stopwords = [] 
  
for w in tokenized_word: 
    if w not in stop_words: 
        tokenized_word_without_Stopwords.append(w) 
  
print(tokenized_word_without_Stopwords) 

# tokenized_word_without_Stopwords = map(lambda w : w if not w in stop_words, tokenized_word)
# print(tokenized_word_without_Stopwords)


# In[84]:


print ("There are {} words in all of the review comments.".format(len(tokenized_word_without_Stopwords)))


# # Frequency Distribution

# In[80]:


from nltk.probability import FreqDist
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[81]:


# Frequency Distribution Plot
fdist = FreqDist(tokenized_word_without_Stopwords)
print(fdist)
fdist.plot(50,cumulative=False) #plot 50 high frequency words
plt.figure(figsize=(50,50))
plt.show()


# # POS tagging (Parts of Speech)

# In[38]:


import nltk
#nltk.download('averaged_perceptron_tagger')


# In[116]:


tagged = nltk.pos_tag(tokenized_word_without_Stopwords) #need to use split()
tagged[0:20]


# Some examples are as below:
# 
# Abbreviation	Meaning
# CC	coordinating conjunction
# CD	cardinal digit
# DT	determiner
# EX	existential there
# FW	foreign word
# IN	preposition/subordinating conjunction
# JJ	adjective (large)
# JJR	adjective, comparative (larger)
# JJS	adjective, superlative (largest)
# LS	list market
# MD	modal (could, will)
# NN	noun, singular (cat, tree)
# NNS	noun plural (desks)
# NNP	proper noun, singular (sarah)
# NNPS	proper noun, plural (indians or americans)
# PDT	predeterminer (all, both, half)
# POS	possessive ending (parent\ 's)
# PRP	personal pronoun (hers, herself, him,himself)
# PRP$	possessive pronoun (her, his, mine, my, our )
# RB	adverb (occasionally, swiftly)
# RBR	adverb, comparative (greater)
# RBS	adverb, superlative (biggest)
# RP	particle (about)
# TO	infinite marker (to)
# UH	interjection (goodbye)
# VB	verb (ask)
# VBG	verb gerund (judging)
# VBD	verb past tense (pleaded)
# VBN	verb past participle (reunified)
# VBP	verb, present tense not 3rd person singular(wrap)
# VBZ	verb, present tense with 3rd person singular (bases)
# WDT	wh-determiner (that, what)
# WP	wh- pronoun (who)
# WRB	wh- adverb (how)
# 
# POS tagger is used to assign grammatical information of each word of the sentence.

# # Create word cloud

# In[72]:


import numpy as np
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


# In[113]:


#Convert tokenized list of words to string to be passed to WordCloud.generate() method
text=' '.join(tokenized_word_without_Stopwords)
type(text)
#str
print (text)


# In[114]:


# Generate a word cloud image
wordcloud = WordCloud(max_font_size=50,background_color="white").generate(text)

# Display the generated image:
# the matplotlib way:
plt.figure(figsize=(30,30))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

wordcloud.to_file("C:/Users/jkadhak/Python Scripts/wordcloud.png")


# From the word cloud we infer:
# 1. This Thermostat might be widely used for freezing beers.
# 2. There are lot of positive words like "great", "simple", "great" ,"easy", "better", "best"
# Since we words like "works", "working" ,"worked" and "work", which essentially mean "work"

# # VADER Sentiment Analysis

# VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon and rule-based sentiment analysis tool that is specifically attuned to sentiments expressed in social media. VADER uses a combination of A sentiment lexicon is a list of lexical features (e.g., words) which are generally labelled according to their semantic orientation as either positive or negative.
# 
# VADER has been found to be quite successful when dealing with social media texts, NY Times editorials, movie reviews, and product reviews. This is because VADER not only tells about the Positivity and Negativity score but also tells us about how positive or negative a sentiment is.
# 
# It is fully open-sourced under the MIT License. The developers of VADER have used Amazon’s Mechanical Turk to get most of their ratings.

# Advantages of using VADER:
# 
# VADER has a lot of advantages over traditional methods of Sentiment Analysis, including:
# 
# It works exceedingly well on social media type text, yet readily generalizes to multiple domains
# It doesn’t require any training data but is constructed from a generalizable, valence-based, human-curated gold standard sentiment lexicon
# It is fast enough to be used online with streaming data, and
# It does not severely suffer from a speed-performance tradeoff.
# The source of this article is a very easy to read paper published by the creaters of VADER library.

# In[40]:


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# Use tokenised sentences to check the sentiment weights associated with each tokenized sentence in reviews.

# In[52]:


sid = SentimentIntensityAnalyzer()
for sentence in tokenized_sent:
    print(sentence)
    ss = sid.polarity_scores(sentence)
    for k in sorted(ss):
        print('{0}: {1}, '.format(k, ss[k]), end='')
    print('\n')


# The Positive, Negative and Neutral scores represent the proportion of text that falls in these categories. This means our sentence was rated as 32% Positive, 68% Neutral and 0% Negative. Hence all these should add up to 1.
# 
# The Compound score is a metric that calculates the sum of all the lexicon ratings which have been normalized between -1(most extreme negative) and +1 (most extreme positive). 
# 
# positive sentiment: compound score>=0.05
# neutral sentiment:  compound score>-0.05 or compound score<0.05
# negative sentiment: compound score<=-0.05
# 

# Let's calculate average polarity score for each of the review comments in Review dataframe. Below mentioned function can be used to get the Polarity,Positive, Negative and Neutral scores  of each review in Dataframe. We have then categorised 
# if polarity is < -0.1 then the review has Negative Sentiment .
# if polarity is >  0.1 then the review has POsitive Sentiment .

# In[50]:


def analyze_sentiment(df):
    sentiments = []
    sid = SentimentIntensityAnalyzer()
    for i in range(df.shape[0]):
        if i % 100 == 0:
            print(i)
        line = df['Review'].iloc[i]
        sentiment = sid.polarity_scores(line)
        sentiments.append([sentiment['neg'], sentiment['pos'],
                           sentiment['neu'], sentiment['compound']])
    df[['neg', 'pos', 'neu', 'compound']] = pd.DataFrame(sentiments)
    df['Negative'] = df['compound'] < -0.1
    df['Positive'] = df['compound'] > 0.1
    return df 


# Write each review polarity back to Excel sheet.

# In[56]:


analyze_sentiment(review)
review.to_csv("AmazonReviewSentimentAnalysis.csv")
review.head()


# Inferences: 
# 1. Out of 611 reviews 49 has negative polarity and 562 has Positive.
# 2.We can now create word cloud for the reviews marked with negative polarity. This we we can further understand what are the customers taking about. Are ther issues with the controller,the customer service or issue with warranty period etc.

# In[69]:


# filter reviews with Negative=True
NegativeReviewsOnly= review.loc[review['Negative'] == True]
len(NegativeReviewsOnly)
#type(df)


# In[70]:


NegativeReviewText=" ".join(review for review in NegativeReviewsOnly['Review_only_text'])
print(NegativeReviewText)


# Create Word cloud of negative comments.

# In[83]:


# Generate a word cloud image exclusing stop words
stop_words.update(["freezer","work","control","product","work","working","works"])
wordcloud = WordCloud(stopwords=stop_words,max_font_size=50,background_color="white").generate(NegativeReviewText)

# Display the generated image:
# the matplotlib way:
plt.figure(figsize=(30,30))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

wordcloud.to_file("C:/Users/jkadhak/Python Scripts/NegativeWordCloud.png")


# Inferences: 
# 1. Form the above word cloud we see the words like instruction, Manual, ordered, purchased, user direction,contact,service.Meaning there can be some complains about user manual/instruction with the controller or there can be some issues with purchasing cycle or customer care service. Business can now further look into these for process improvements.
# 2. Also we see some words like settings, Shut, plug, probe,Power option, interface. These can point to some unsatisfactory customer with the design or parts of the thermostat.
# 
# Business now has some improvement areas to look into further.
