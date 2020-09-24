import tweepy
import csv
import pandas as pd
import datetime

consumer_key  = 'rHNuLd1AdysA6MTcB3vViFNjx'
consumer_secret  = 'PyZUOO1SDyq77WcBVOZyZ0WJDONWv74gJdImkZOADvH9AUKPCh'
access_token  = '92525746-rfiCQ9w9opvhLmt4GoItnMEJNTbUV003lYcz3mrjV'
access_token_secret  = 'YfnFFS6rRNIQlrdf3JD9iKu6Apqvi7uYPJRqohJRbN7Kw'


auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)


df = pd.DataFrame({"Dates":[], 
                    "Text":[]})
tweets = []

for tweet in tweepy.Cursor(api.search,q="spacex stock",count=100,
                           lang="en",since="2020-01-01").items():
    print (tweet.created_at, tweet.text)


    df_temp = pd.DataFrame({"Dates":[tweet.created_at], 
                    "Text":[tweet.text]})

    df=df.append(df_temp, ignore_index = True)
print(df)

df.to_csv('spacex.csv')



