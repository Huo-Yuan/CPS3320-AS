import nltk
import pandas as pd
import numpy as np
import re
nltk.download('stopwords')
from nltk.corpus import stopwords
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

stop = stopwords.words('english')

ratings = pd.read_csv(r"C:\Users\user\Desktop\AMZN.csv")
df=ratings.loc[:,['date','time','tweet','retweets_count','cashtags']]
Tweet = ratings['tweet']
Tweet.head()
df['change'] = df['cashtags'].map(lambda x:re.sub(r',','',x))
df['length'] = df['cashtags'].str.len()
df['change_length'] = df['change'].str.len()
df['cash_number'] = df['length']- df['change_length'] + 1
print(df)
df = df.loc[df['cash_number'] <= 3]
df = df.loc[:,['date','time','tweet','retweets_count']]
#
# df["tweet"] = df['tweet'].str.replace('[^\w\s]','')
# df["tweet"] = df["tweet"].str.lower().str.split()

df['tweet'] = df['tweet'].apply(lambda x: ' '.join([word for word in x.lower().split() if word not in (stop)]))
df.to_csv("hh.csv")
print(df)
def sentimentScore(Tweet):
    analyzer = SentimentIntensityAnalyzer()
    results = []
    for sentence in Tweet:
        vs = analyzer.polarity_scores(sentence)

        #print("{: <65} {}".format(sentence, str(vs)))
        #NOTE! I blocked the second print command so the sentences are
        #left out in the cell below, purely for clarity reasons
        results.append(vs)
    return results
df_results = pd.DataFrame(sentimentScore(Tweet))
df_tweets = pd.merge(df, df_results, left_index=True, right_index=True)

df_tweets['weighted score'] = df_tweets['compound']*(df_tweets['retweets_count']+1)
df_tweets = df_tweets[(df_tweets[['compound']] != 0).all(axis=1)]

df_tweets=df_tweets.loc[:,['date','time','retweets_count','weighted score']]
print(df_tweets)
df_tweets['date'] = pd.to_datetime(df_tweets['date'])
df_tweets.sort_values(by='date')
df_daily_sum=df_tweets.groupby('date').sum()
df_daily_sum.drop('retweets_count',axis = 1,inplace = True)

df_daily_sum.rename(columns={'weighted score': 'Sumscore'}, inplace=True)
df_daily_sum.to_csv(r'C:\Users\user\Desktop\Dailysum.csv')
print(df_daily_sum)






