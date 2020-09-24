import numpy as np 
import pandas as pd
import string
from tqdm import tqdm
import math,nltk
import re
import time
from sklearn import feature_extraction
from textblob import TextBlob 
import numpy
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import Dense
from keras import losses
from keras import activations
import sklearn
from keras.layers import LSTM
from matplotlib import pyplot as plt
from tkinter import *
from PIL import Image, ImageTk
from tkinter import messagebox



_wnl = nltk.WordNetLemmatizer()

def normalize_word(w):
    return _wnl.lemmatize(w).lower()

def get_tokenized_lemmas(s):
    return [normalize_word(t) for t in nltk.word_tokenize(s)]

def clean(text):
    # Cleans a string: Lowercasing, trimming, removing non-alphanumeric
    text = text.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation),''))
    return " ".join(re.findall(r'\w+', text, flags=re.UNICODE)).lower()

def remove_stopwords(l):
    # Removes stopwords from a list of tokens
    return [w for w in l if w not in feature_extraction.text.ENGLISH_STOP_WORDS]
def join_tok(text):
    return " ".join(text).lower()

def process(texts):
    lst=[]
    for text in tqdm(texts):
        clean_text= clean(text)
        tok_text= get_tokenized_lemmas(clean_text)
        remov_stp= remove_stopwords(tok_text)
        lst.append(join_tok(remov_stp))
    return lst

def senti_polarity(tweets):
    senti_score=[]
    for tweet in tweets:
        analysis = TextBlob(tweet)
        senti_score.append(analysis.sentiment.polarity)
    return senti_score

def get_tweet_sentiment(tweet): 

        analysis = TextBlob(tweet)
        if analysis.sentiment.polarity > 0: 
            return 'positive'
        elif analysis.sentiment.polarity == 0: 
            return 'neutral'
        else: 
            return 'negative'

root = Tk()  # Main window 
f = Frame(root)
frame1 = Frame(root)
frame2 = Frame(root)
frame3 = Frame(root)
root.title("Stock Market Analysis With Sentiment Analysis On Twitter")
root.geometry("1080x720")

canvas = Canvas(width=1080, height=250)
canvas.pack()
filename=('stock1.png')
load = Image.open(filename)
load = load.resize((1800, 250), Image.ANTIALIAS)
render = ImageTk.PhotoImage(load)
img = Label(image=render)
img.image = render
#photo = PhotoImage(file='landscape.png')
load = Image.open(filename)
img.place(x=1, y=1)
#canvas.create_image(-80, -80, image=img, anchor=NW)


root.configure(background='Green')
scrollbar = Scrollbar(root)
scrollbar.pack(side=RIGHT, fill=Y)

firstname = StringVar()  # Declaration of all variables
lastname = StringVar()
id = StringVar()
dept = StringVar()
designation = StringVar()
remove_firstname = StringVar()
remove_lastname = StringVar()
searchfirstname = StringVar()
sevierity = StringVar()
sheet_data = []
row_data = []





def add_entries():  # to append all data and add entries on click the button
    a = " "
    f = sevierity.get()
    f1 = f.lower()
    l = lastname.get()
    l1 = l.lower()
    d = dept.get()
    d1 = d.lower()
    de = designation.get()
    de1 = de.lower()
    
    list1 = list(a)
    list1.append(f1)
    list1.append(l1)
    list1.append(d1)
    list1.append(de1)



def visualizations():
    import seaborn as sns
    sns.set_style('whitegrid')
    plt.style.use("fivethirtyeight")
    from datetime import datetime

    name=sevierity.get()
    filenames=name + '_stock.csv'
    data=pd.read_csv(filenames)
    #Adj Close
    plt.figure(figsize=(12, 8))
    plt.subplots_adjust(hspace=0.77,bottom=0.28)




    plt.subplot(2, 2, 1)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index(data['Date'], inplace=True)
    data['Adj Close'].plot()
    plt.ylabel('Adj Close')
    plt.xlabel(None)
    plt.title(f"{name}")


    plt.subplot(2, 2, 2)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index(data['Date'], inplace=True)
    data['Close'].plot()
    plt.ylabel('Close')
    plt.xlabel(None)
    plt.title(f"{name}")



    plt.subplot(2, 2, 3)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index(data['Date'], inplace=True)
    data['High'].plot()
    plt.ylabel('High')
    plt.xlabel(None)
    plt.title(f"{name}")
    plt.show()



def senti_graph():
    name=sevierity.get()
    filenames=name + '.csv'
    df = pd.read_csv(filenames)
    # ptweets=[]
    # ntweets=[]

    titles=  process(df['Text'])

    df['Text']=titles


    ptweets =[tweet for tweet in titles if get_tweet_sentiment(tweet) == 'positive']
    a=100*len(ptweets)/len(df)

    ntweets =[tweet for tweet in titles if get_tweet_sentiment(tweet) == 'negative']
    b=100*len(ptweets)/len(df)

    c=100-a-b
    b=100*len(ptweets)/len(df)


    e3.delete(0, END) #deletes the current value
    e3.insert(0, a)
    e4.delete(0, END) #deletes the current value
    e4.insert(0, b)
    e5.delete(0, END) #deletes the current value
    e5.insert(0, c)

    #pie chart
    labels = 'Possitive', 'Neutral', 'Negative'
    sizes = [a,b,c]
    explode = (0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'possitive')

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.show()




def click():
    name=sevierity.get()
    filenames=name + '.csv'
    df = pd.read_csv(filenames)

    titles=  process(df['Text'])

    df['Text']=titles
    df['sentiment']=senti_polarity(titles)

    print(df)

    df = df.drop(['Text'],1)
    df['Date'] = pd.to_datetime(df['Date'])
    group = df.groupby('Date')
    sentiment_avg = group['sentiment'].mean()

    df_stock=pd.read_csv('Stocks_dataset/'+name+'_stock.csv')
    df_stock['sentiment_polarity']=sentiment_avg.values
    print(df_stock)

    X=df_stock[['sentiment_polarity','Open','High','Low','Adj Close']]
    Y=df_stock[['Close']]

    # X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.33)

    length=(len(df_stock) / 100)
    length=round(length * 80)

    X_train=X[0:length] 
    X_test=X[length:]
    Y_train=Y[0:length] 
    Y_test=Y[length:]


    from sklearn import preprocessing
    min_max_scalar=preprocessing.MinMaxScaler()
    X_train=min_max_scalar.fit_transform(X_train)
    X_test=min_max_scalar.fit_transform(X_test)
    Y_train=min_max_scalar.fit_transform(Y_train)
    Y_test=min_max_scalar.fit_transform(Y_test)

    #model
    model=Sequential()
    model.add(Dense(5,activation=activations.sigmoid,input_shape=(5,)))
    model.add(Dense(100,activation=activations.sigmoid))
    model.add(Dense(100,activation=activations.sigmoid))
    model.add(Dense(100,activation=activations.sigmoid))
    model.add(Dense(100,activation=activations.sigmoid))
    model.add(Dense(100,activation=activations.sigmoid))
    model.add(Dense(1,activation=activations.sigmoid))

    model.compile(optimizer='adam',loss=losses.mean_absolute_error)

    model.fit(X_train,Y_train,verbose=2,epochs=1000)

    y_pred=model.predict(X_test)

    print(r2_score(Y_test,y_pred))

    # print(min_max_scalar.inverse_transform(y_pred))

    y_pred=min_max_scalar.inverse_transform(y_pred)
    Y_test=min_max_scalar.inverse_transform(Y_test)

    date_list=df_stock['Date']
    pred=y_pred[-7:]
    orig=Y_test[-7:]
    dates=date_list[-7:]
    date_df=pd.DataFrame(dates,columns=['Date'])



    # Visualising the results
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(25,15), dpi=80, facecolor='w', edgecolor='k')
    ax = plt.gca()  
    plt.plot(orig, color = 'red', label = 'Real Stock Price')
    plt.plot(pred, color = 'blue', label = 'Predicted Stock Price')
    plt.title('Stock Price Prediction', fontsize=40)
    date_df.set_index('Date', inplace= True)
    date_df = date_df.reset_index()
    x=date_df.index
    labels = date_df['Date']
    plt.xticks(x, labels, rotation = 'vertical')
    for tick in ax.xaxis.get_major_ticks():
                tick.label1.set_fontsize(18)
    for tick in ax.yaxis.get_major_ticks(): 
                tick.label1.set_fontsize(18)
    plt.xlabel('Time', fontsize=40)
    plt.ylabel('Stock Price', fontsize=40)
    plt.legend(loc=2, prop={'size': 25})
    plt.show()



def clear_all():  # for clearing the entry widgets
    frame1.pack_forget()
    frame2.pack_forget()
    frame3.pack_forget()


label1 = Label(root, text="Stock Market Prediction With Sentiment Analysis")
label1.config(font=('Italic', 18, 'bold'), justify=CENTER, background="Yellow", fg="Red", anchor="center")
label1.pack(fill=X)


frame2.pack_forget()
frame3.pack_forget()


satisfaction_level = Label(frame2, text="Enter Query Text: ", bg="red", fg="Black")
satisfaction_level.grid(row=1, column=1, padx=10)
sevierity.set("Select Stock")
e1 = OptionMenu(frame2, sevierity, "Select Option", "AAPL", "AMZN", "FB", "PFE", "TYO" ,"XPO")
e1.grid(row=1, column=2, padx=10)


button5 = Button(frame2, text="Submit", command=click)
button5.grid(row=1, column=3, pady=10,padx=10)


sentilabel = Label(frame2, text="View Sentiment: ", bg="red", fg="Black")
sentilabel.grid(row=2, column=1, padx=10)

button2 = Button(frame2, text="Sentiment",command=senti_graph)
button2.grid(row=2, column=2, pady=10,padx=10)


visualabel = Label(frame2, text="View Visualization: ", bg="red", fg="Black")
visualabel.grid(row=3, column=1, padx=10)

button2 = Button(frame2, text="Visualization",command=visualizations)
button2.grid(row=3, column=2, pady=10,padx=10)


label1 = Label(frame1, text="Possitive Tweets ", bg="red", fg="Black")
label1.grid(row=4, column=1, padx=10, pady=10)
e3 = Entry(frame1)
e3.grid(row=5, column=1, padx=10, pady=10)

label2 = Label(frame1, text="Negative Tweets ", bg="red", fg="Black")
label2.grid(row=4, column=2, padx=10, pady=10)
e4 = Entry(frame1)
e4.grid(row=5, column=2, padx=10, pady=10)

label3 = Label(frame1, text="Neutral Tweets ", bg="red", fg="Black")
label3.grid(row=4, column=3, padx=10, pady=10)
e5 = Entry(frame1)
e5.grid(row=5, column=3, padx=10, pady=10)


frame2.configure(background="Red")
frame2.pack(pady=10)

frame1.configure(background="Red")
frame1.pack(pady=10)

root.mainloop()






















