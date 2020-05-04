from flask import Flask, render_template, request, url_for, Markup
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import io, base64, os, json, re, math

plt.style.use('bmh') 

filename=''
future_days=25


app = Flask(__name__)

@app.before_first_request
def import_yahoofinance():
    from pandas_datareader import data as pdr
    from datetime import date
    import yfinance as yf
    import datetime

    yf.pdr_override()

    # Tickers list
    # We can add and delete any ticker from the list to get desired ticker live data
    today = date.today()
    start_date= '2017/01/01'
    end_date='2020/04/24'

    start_date=datetime.datetime.strptime(start_date, '%Y/%m/%d')
    end_date=datetime.datetime.strptime(end_date, '%Y/%m/%d')

    print(start_date)
    #ticker_list=['DJIA', 'DOW', 'LB', 'EXPE', 'PXD', 'MCH', 'CRM', 'JEC' , 'NRG', 'HFC', 'NOW']
    ticker_list=['DJIA','DOW','LB','EXPE','PXD']
    # We can get data by our choice by giving days bracket
    files=[]
    def getData(ticker):
        print(ticker)
        data = pdr.get_data_yahoo(ticker, start=start_date, end=today)
        dataname= ticker+'_'+str(today)
        files.append(dataname)
        SaveData(data, dataname)
    
    # Create a data folder in your current dir.
    def SaveData(df, filename):
        df.to_csv(filename+'.csv')


        
#This loop will iterate over ticker list, will pass one ticker to get data, and save that data as file.
    for tik in ticker_list:
        getData(tik)
    #for i in range(0,11):
    #    df1= pd.read_csv('./data/'+ str(files[i])+'.csv')
    df1= pd.read_csv(str(files[0])+'.csv')
    #print (df1.tail())
    return str(files[0])+'.csv'

def import_yahoofinance2():
    # import stock_info module from yahoo_fin
    from yahoo_fin import stock_info as si

    # get live price of Apple
    si.get_live_price("aapl")
    
    # or Amazon
    si.get_live_price("amzn")
    
    # or any other ticker
    si.get_live_price(ticker)

    # get quote table back as a data frame
    si.get_quote_table("aapl", dict_result = False)
    
    # or get it back as a dictionary (default)
    si.get_quote_table("aapl")

    si.get_top_crypto()
    # get most active stocks on the day
    si.get_day_most_active()
    
    # get biggest gainers
    si.get_day_gainers()
    
    # get worst performers
    si.get_day_losers()


@app.route("/", methods=['GET','POST'])
def GetForecast():

   # import tk as ttk
   # import tkinter as tk
   # from tkinter import *


    choices1='DJIA'
    
    
   # root = tk.Tk()
   # root.title("Tk dropdown example")

    # Add a grid
    
   # mainframe = Frame(root)
   # mainframe.grid(column=0,row=0, sticky=(N,W,E,S) )
   # mainframe.columnconfigure(0, weight = 1)
   # mainframe.rowconfigure(0, weight = 1)
   # mainframe.pack(pady = 100, padx = 100)

    # Create a Tkinter variable
    #tkvar = StringVar(root)

    # Dictionary with options
    choices = ['DJIA','DOW','LB','EXPE','PXD']
   
   # tkvar.set('DJIA') # set the default option

   # popupMenu = OptionMenu(mainframe, tkvar, *choices)
   # Label(mainframe, text="Choose a ticker").grid(row = 1, column = 1)
   # popupMenu.grid(row = 2, column =1)

    # on change dropdown value
   # def change_dropdown(*args):
    # link function to change dropdown
   #     tkvar.trace('w', change_dropdown)



    #if request.method == 'POST':
    linewidth=5
    #    choices1 = int(request.form['choices1'])

    #root.mainloop()
    
    #xx=tkvar.get()
    #print(xx) 

    filename=import_yahoofinance()
    print(filename)
    df= pd.read_csv(filename)
    print(df.head(6))
    print(df.shape)

    plt.figure(figsize=[16,8])
    plt.title(choices1)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.plot(df['Date'],df['Close'])
    #plt.show()

        
    df2=df[['Close']]
    df2['Prediction']=df2[['Close']].shift(-future_days)
    #print(df2.head())

    X=np.array(df2.drop(['Prediction'],1))[:-future_days]
   # print(X,X.shape)
    Y=np.array(df2['Prediction'])[:-future_days]
    #print(Y,Y.shape)
    x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25)


    #RegressorModel

    tree=DecisionTreeRegressor().fit(x_train,y_train)

    #Linear Regresion Model

    lr=LinearRegression().fit(x_train,y_train)


    x_future=df2.drop(['Prediction'],1)[:-future_days]
    x_future=x_future.tail(future_days)

    x_future=np.array(x_future)
    #print(x_future)

    tree_prediction=tree.predict(x_future)
    #print(tree_prediction)

    lr_prediction=lr.predict(x_future)
    #print(lr_prediction)

    #----------------------------------------------- plot predictions -----

    predictions = tree_prediction

    valid=df2[X.shape[0]:]
    valid['Predictions']=predictions
    plt.figure(figsize=(16,8))
    plt.title('Model Tree Prediction')
    plt.xlabel('Days')
    plt.ylabel('Close Price')
    plt.plot(df2['Close'])
    plt.plot(valid[['Close','Predictions']])
    plt.legend(['Orig', 'Val','Predict'])
    #plt.show()

    img = io.BytesIO()
    plt.savefig(img,format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()



    predictions = lr_prediction

    valid=df2[X.shape[0]:]
    valid['Predictions']=predictions
    plt.figure(figsize=(16,8))
    plt.title('Model Tree Prediction')
    plt.xlabel('Days')
    plt.ylabel('Close Price')
    plt.plot(df2['Close'])
    plt.plot(valid[['Close','Predictions']])
    plt.legend(['Orig', 'Val','Predict'])
    #plt.show()

    img = io.BytesIO()
    plt.savefig(img,format='png')
    img.seek(0)
    plot_url2 = base64.b64encode(img.getvalue()).decode()


    return render_template('forecast.html',
        forecast_plot=Markup('<img src="data:image/png;base64,{}" style="width:80%;">'.format(plot_url)),choices1 =choices1,default_choices=choices1,forecast_plot2=Markup('<img src="data:image/png;base64,{}" style="width:80%;">'.format(plot_url2)))

if __name__=='__main__':
    app.run(debug=True)


