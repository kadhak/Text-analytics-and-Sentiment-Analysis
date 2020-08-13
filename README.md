# Text-analytics-and-Sentiment-Analysis
**Text analytics and Sentiment Analysis for Product review comments.**  
**Input :**  
1. Use "AmazonComments.csv" which contains the review comments fetched from amazon using some R code.  
**Output :**
1. WordCloud
2. CSV file - Sentiment polarity of each of the review comment."AmazonReviewSentimentAnalysis.csv"
3. We observe that out of 611 comments 49 had negative polarity and 562 had positive or neutral polarity.
4. Form the negative word cloud we see the words like instruction, Manual, ordered, purchased, user direction,contact,service.Meaning there can be some complains about user manual/instruction with the controller or there can be some issues with purchasing cycle or customer care service. Business can now further look into these for process improvements.
5.Also we see some words like settings, Shut, plug, probe,Power option, interface. These can point to some unsatisfactory customer with the design or parts of the thermostat.
6. Business now has some improvement areas to look into further.
            
**Sentiment Analysis from twitter- Textblob and NLTK VaderSentimentAnalyser**
**Steps :**  
1. Get tweets from twitter api with the help of keys generated through twitter developer account and tweepy package in python. You can find details steps [here](https://towardsdatascience.com/analysis-of-tweets-about-the-joker-2019-film-in-python-df9996aa5fb1).
2. Clean the tweets using tweet preprocessor package.
3. Add addition cleaning steps for emojis.
4. Seperate hastags and mentions from the tweet text.
5. Using Textblob package get sentiment score and subjectivity of each tweet.
6. Using NLTK VaderSentimentAnalyser get compound score for each tweet.
7. Store the original tweets and sentiments in csv.
8. User can pass one or more keywords to fetch the tweets to twitter api.

![Sample output](https://github.com/kadhak/Text-analytics-and-Sentiment-Analysis/blob/master/IphoneSentimentScores.PNG)

References:  
1. https://towardsdatascience.com/extracting-twitter-data-pre-processing-and-sentiment-analysis-using-python-3-0-7192bd8b47cf
