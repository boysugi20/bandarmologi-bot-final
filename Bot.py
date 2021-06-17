import os, subprocess, atexit
import sys
import datetime
import time
import random

from tabulate import tabulate
from sty import fg, bg
import yfinance as yf

from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec
from bs4 import BeautifulSoup

import concurrent.futures

from pathlib import Path
import pandas as pd
import numpy as np

def testPlot(y):
    import plotext as plx

    plx.clear_plot()

    y.reverse()

    x = []
    count = 1
    new_y = []
    for i in range(len(y)):
        y[i] = y[i].replace(",", "")
        count = count + round(float(y[i]), 2)
        new_y.append(count)
        x.append(i)

    x_array = np.asarray(x)
    y_array = np.asarray(new_y)

    zero = []
    for i in range(len(x_array)):
        zero.append(0)
    zero_array = np.asarray(zero)
    
    # print(y_array)
    plx.scatter(x_array, zero_array, rows = 20, cols = len(x_array)*2, point_marker = '-', line_marker = '-' , point_color = 'yellow', line = True, ticks = False)
    plx.scatter(x_array, y_array, rows = 20, cols = len(x_array)*2, point_marker = '•', ticks = False)
    plx.show()
    print('\n')

def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = str(fg.green + '$' + fg.rs) * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s %s\r' % (bar, percents, '%', status))
    sys.stdout.flush()

def welcomeMessages():
    os.system('cls')
    print(  fg.green +
            "\n"+
            " $$\                                 $$\                                             $$\                     $$\ \n"+
            " $$ |                                $$ |                                            $$ |                    \__|\n"+
            " $$$$$$$\   $$$$$$\  $$$$$$$\   $$$$$$$ | $$$$$$\   $$$$$$\  $$$$$$\$$$$\   $$$$$$\  $$ | $$$$$$\   $$$$$$\  $$\ \n"+
            " $$  __$$\  \____$$\ $$  __$$\ $$  __$$ | \____$$\ $$  __$$\ $$  _$$  _$$\ $$  __$$\ $$ |$$  __$$\ $$  __$$\ $$ |\n"+
            " $$ |  $$ | $$$$$$$ |$$ |  $$ |$$ /  $$ | $$$$$$$ |$$ |  \__|$$ / $$ / $$ |$$ /  $$ |$$ |$$ /  $$ |$$ /  $$ |$$ |\n"+
            " $$ |  $$ |$$  __$$ |$$ |  $$ |$$ |  $$ |$$  __$$ |$$ |      $$ | $$ | $$ |$$ |  $$ |$$ |$$ |  $$ |$$ |  $$ |$$ |\n"+
            " $$$$$$$  |\$$$$$$$ |$$ |  $$ |\$$$$$$$ |\$$$$$$$ |$$ |      $$ | $$ | $$ |\$$$$$$  |$$ |\$$$$$$  |\$$$$$$$ |$$ |\n"+
            " \_______/  \_______|\__|  \__| \_______| \_______|\__|      \__| \__| \__| \______/ \__| \______/  \____$$ |\__|\n"+
            "                                                                                                   $$\   $$ |    \n"+
            "                                                                                                   \$$$$$$  |    \n"+
            "                                                                                                    \______/     \n"+
            fg.rs +
            fg.yellow + "\nBot command list :\n" +fg.rs+
            "1. Input ticker to view weekly broker summary of the selected ticker\n"+
            "2. Input \"" + fg.red + "view" + fg.rs +"\" to view all the available ticker on daily_data folder\n"+
            "3. Input \"" + fg.red + "top(space)n(space)dd-mm-yyyy" + fg.rs + "\" where n is the net ratio used,\n   to view top acc/dist on that date[ex: top 3 10-10-2020]\n"+
            "4. Input \"" + fg.red + "recommend(space)n" + fg.rs + "\" where n is the net ratio used, to view top acc/dist\n   weekly all ticker. [ex: recommend 3]\n"+
            "5. Input \"" + fg.red + "history(space)ticker(space)n" + fg.rs + "\" where n is the net ratio used, to\n   view acc/dist history of selected ticker. [ex: history antm 3]\n"+
            "6. Input \"" + fg.red + "daily(space)ticker(space)dd-mm-yyyy)" + fg.rs + "\" where dd-mm-yyyy is the date,\n   to view broker summary of selected ticker on selected date.\n   [ex: daily antm 20-07-2000]\n" +
            "7. Input \"" + fg.red + "/download" + fg.rs + "\" to download broker summary data\n" + 
            "8. Input \"" + fg.red + "spike(space)n" + fg.rs + "\", where n is number of days backwards, to show tickers with\n   volume spike. [ex: spike 10]\n" +
            "9. Input \"" + fg.red + "/q" + fg.rs + "\" to exit the application\n"
    )

def prettyIndex(x):
    y = []
    for i in x:
        if(i < 10):
            y.append(fg.magenta + str("0" + str(i)) + fg.rs)
        else:
            y.append(fg.magenta + str(i) + fg.rs)
    return y

def viewList():
    os.system('cls')

    #Get all the different ticker in all daily folder
    tickers = []
    dateListTemp = os.listdir(path + '\daily_data/')
    for date in dateListTemp:
        tickerList = os.listdir( path + '\daily_data\\' + date + "/")
        for ticker in tickerList:
            if(ticker[:4] not in tickers):
                tickers.append(ticker[:4])

    #Remove file extension
    newTickers = [x[:4] for x in tickers]

    print("Downloaded ticker(s) :")
    no = 1
    for x in newTickers:
        if(no < 10):
             num = "0" + str(no)
        else:
             num = str(no)
        print(num + ". " + x)
        no = no + 1
    print("\n")

def getPath():
    path = Path.cwd()
    return str(path)

def findBroker(df_buy, df_sell, broker):
    Broker_Location = "Not found"
    #Found on buy
    if(df_buy[df_buy['NBY'].str.contains(str(broker))].empty == False):
        temp = int(df_buy[df_buy['NBY']==str(broker)].index.values[0]) + 1
        Broker_Location = str(fg.red + "TOP " + str(temp) + " BUY" + fg.rs)
    #Found on sell
    if(df_sell[df_sell['NSL'].str.contains(str(broker))].empty == False):
        temp = int(df_sell[df_sell['NSL']==str(broker)].index.values[0]) + 1
        Broker_Location = str(fg.green + "TOP " + str(temp) + " SELL" + fg.rs)
        
    return Broker_Location

def newDataFrame(link, header_number, column_array):
    new_data_frame = pd.read_excel(link, header=header_number, usecols=column_array)
    return new_data_frame

def reformatExcel(data_frame, column_name):
    data_frame[column_name].fillna("0", inplace=True)

    #Manage all M, B, T
    data_frame = data_frame.astype('str')
    data_frame = data_frame.apply(lambda x: x.str.replace(',', ''))
    data_frame[column_name] = data_frame[column_name].apply(lambda x: x.replace('.', '').replace('K', '00') if x.find(".") != -1 and x.find("K") != -1 else x.replace('K', '000'))
    data_frame[column_name] = data_frame[column_name].apply(lambda x: x.replace('.', '').replace('M', '00000') if x.find(".") != -1 and x.find("M") != -1 else x.replace('M', '000000'))
    data_frame[column_name] = data_frame[column_name].apply(lambda x: x.replace('.', '').replace('B', '00000000') if x.find(".") != -1 and x.find("B") != -1 else x.replace('B', '000000000'))
    data_frame[column_name] = data_frame[column_name].apply(lambda x: x.replace('.', '').replace('T', '00000000000') if x.find(".") != -1 and x.find("T") != -1 else x.replace('T', '000000000000'))
    data_frame[column_name] = data_frame[column_name].astype(float)
    data_frame.sort_values(by=column_name, ascending=False)

    return data_frame

def calculateNetVolumeLot(data_frame, column_name):
    total_lot = pd.to_numeric(data_frame[column_name]).sum()
    return total_lot

def calculateNetValue(data_frame, column_name):
    data_frame.fillna("0", inplace=True)
    valueArray = data_frame[column_name].tolist()
    sum = 0
    for x in valueArray:
        if(x == 'nan'):
            temp = 0
        else:
            temp = int(x)
        sum = sum + temp
    return sum

def calculateTop(total_broker, data_frame_buy, buy_column_name, data_frame_sell, sell_column_name):
    colouredA = str(fg.green + "ACCUMULATION" + fg.rs)
    colouredD = str(fg.red + "DISTRIBUTION" + fg.rs)
    status = np.where(data_frame_buy[buy_column_name].head(total_broker).sum() > -(data_frame_sell[sell_column_name].head(total_broker).sum()), str(colouredA), str(colouredD))
    return str(status)

def calculateNetLot(total_broker, data_frame_buy, buy_column_name, data_frame_sell, sell_column_name):
    total_lot = pd.to_numeric(data_frame_buy[buy_column_name].head(total_broker)).sum() + pd.to_numeric(data_frame_sell[sell_column_name].head(total_broker)).sum()
    return total_lot
    
def calculateNetRatio(net_lot, net_volume_lot):
    if(net_volume_lot != 0):
        net_ratio = net_lot  / float(net_volume_lot) * 100
    else:
        net_ratio = -99999
    return net_ratio

def displayNetRatioStatus(net_ratio):
    if net_ratio > 0:
        if net_ratio >30:
            colouredText = fg.green + "BIG ACCUMULATION" + fg.rs
            return colouredText
        elif net_ratio <= 30 and net_ratio > 10:
            colouredText = fg.green + "ACCUMULATION" + fg.rs
            return colouredText
        elif net_ratio <= 10 and net_ratio > 5:
            colouredText = fg.green + "SMALL ACCUMULATION" + fg.rs
            return colouredText
        elif net_ratio <= 5:
            colouredText = fg.yellow + "NETRAL" + fg.rs
            return colouredText
    else:
        if net_ratio >= -5:
            colouredText = fg.yellow + "NETRAL" + fg.rs
            return colouredText
        elif net_ratio >= -10:
            colouredText = fg.red + "SMALL DISTRIBUTION" + fg.rs
            return colouredText
        elif net_ratio >= -30 and net_ratio < -10:
            colouredText = fg.red + "DISTRIBUTION" + fg.rs
            return colouredText
        elif net_ratio <= -30:
            colouredText = fg.red + "BIG DISTRIBUTION" + fg.rs
            return colouredText

def recommendBest(tickers, n):
    netRatioArray = []
    tickerCopy = []
    count = 0
    tempTicker = ''
    
    for x in tickers:
        #Pecah file excel jadi dua (BUY & SELL)
        df_buy = newDataFrame(path + '\\weekly_data\\' + x, 2, [0,1,2])
        df_sell = newDataFrame(path + '\\weekly_data\\' + x, 2, [5,6,7])
        tempTicker = x

        #nama column yang kita mau (sementara maybe(?))
        buy_column_name = list(df_buy.columns)[1] 
        sell_column_name = list(df_sell.columns)[1]
        buy_column_name2 = list(df_buy.columns)[2] 
        sell_column_name2 = list(df_sell.columns)[2]

        #Reformat column dan ubah datatype ke float
        df_buy = reformatExcel(df_buy, buy_column_name)
        df_sell = reformatExcel(df_sell,sell_column_name)
        df_buy = reformatExcel(df_buy, buy_column_name2)
        df_sell = reformatExcel(df_sell,sell_column_name2)

        #Dipakai untuk calculate net ratio
        net_volume_lot = calculateNetVolumeLot(df_buy, buy_column_name)
        net_ratio = calculateNetRatio(calculateNetLot(n, df_buy, buy_column_name, df_sell, sell_column_name),net_volume_lot)
        net_ratio = round(net_ratio, 2)

        if(calculateNetValue(df_buy,"NBVal") > 500000000):
            #Append net ratio
            netRatioArray.append(net_ratio)

            #Change ticker string
            tickerCopy.append(x[:4])
        
        #LoadingBar
        count += 1
        if(count == 1): print("\n")
        progress(count, len(tickers), status='  Calculating...')

    #Create table
    dictTickerRatio = {'Ticker':tickerCopy,'Ratio':netRatioArray}
    df_recommend = pd.DataFrame(dictTickerRatio)

    #Sort
    df_recommend.sort_values(by="Ratio", ascending=False, inplace = True)
    df_recommend.reset_index(drop=True, inplace=True)
    df_recommend.index += 1

    #Colors
    counter = 0
    for row in df_recommend.itertuples(index=True, name='Pandas'):
        x = float(getattr(row, "Ratio"))
        if(x >= 0):
            split_index = counter
        counter += 1

    #Split into two
    split_index += 1
    df1 = df_recommend.iloc[:split_index, :]
    df2 = df_recommend.iloc[split_index:, :]

    df1.columns = ['A_Ticker', 'A_Ratio']
    df2.columns = ['D_Ticker', 'D_Ratio']

    df2 = df2.sort_values(by="D_Ratio", ascending = True)

    df1.reset_index(drop=True, inplace=True)
    df2.reset_index(drop=True, inplace=True)

    #Combine dataframe
    final_result_dataframe = pd.concat([df1,df2], ignore_index = False, axis=1)
    final_result_dataframe.index += 1

    final_result_dataframe.fillna('-', inplace=True)

    #Colors
    for row in final_result_dataframe.itertuples(index=True, name='Pandas'):
        x = getattr(row, "A_Ratio")
        y = getattr(row, "D_Ratio")

        if(x != '-' and type(x) != str):
            if(x < 0):
                final_result_dataframe.replace(to_replace = x , value = str(fg.red + str(x) + fg.rs), inplace = True)
            else:
                final_result_dataframe.replace(to_replace = x , value = str(fg.green + str(x) + fg.rs), inplace = True)
    
        if(y != '-' and type(y) != str):
            if(y < 0):
                final_result_dataframe.replace(to_replace = y , value = str(fg.red + str(y) + fg.rs), inplace = True)
            else:
                final_result_dataframe.replace(to_replace = y , value = str(fg.green + str(y) + fg.rs), inplace = True)

    #Get date range from excel
    date1 = pd.read_excel(path + '\\weekly_data\\' + tempTicker, 'Sheet1', index_col=None, usecols = "D", header = 0, nrows=0)
    date1 = date1.columns.values[0]
    date2 = pd.read_excel(path + '\\weekly_data\\' + tempTicker, 'Sheet1', index_col=None, usecols = "F", header = 0, nrows=0)
    date2 = date2.columns.values[0]

    #Print
    os.system('cls')
    print(fg.yellow + "\nACCUM/DIST (Top " + str(n) + " net ratio) [" + str(date1[0:10]) + ' to ' + str(date2[0:10]) + "]:" + fg.rs + "\n")
    print(tabulate(final_result_dataframe, headers='keys', tablefmt='fancy_grid', stralign="center", disable_numparse=True))
    print("\n\n")

def topDaily(date, n):
    tickers = os.listdir(path + '\\daily_data\\' + date + '\\')

    upload = 'no'

    netRatioArray = []
    tickerCopy = []
    count = 0
    tempTicker = ''
    
    for x in tickers:
        #Pecah file excel jadi dua (BUY & SELL)
        df_buy = newDataFrame(path + '\\daily_data\\' + date + '\\' + x, 2, [0,1,2])
        df_sell = newDataFrame(path + '\\daily_data\\' + date + '\\' + x, 2, [5,6,7])
        tempTicker = x

        #nama column yang kita mau (sementara maybe(?))
        buy_column_name = list(df_buy.columns)[1] 
        sell_column_name = list(df_sell.columns)[1]
        buy_column_name2 = list(df_buy.columns)[2] 
        sell_column_name2 = list(df_sell.columns)[2]

        #Reformat column dan ubah datatype ke float
        df_buy = reformatExcel(df_buy, buy_column_name)
        df_sell = reformatExcel(df_sell,sell_column_name)
        df_buy = reformatExcel(df_buy, buy_column_name2)
        df_sell = reformatExcel(df_sell,sell_column_name2)

        #Dipakai untuk calculate net ratio
        net_volume_lot = calculateNetVolumeLot(df_buy, buy_column_name)
        net_ratio = calculateNetRatio(calculateNetLot(n, df_buy, buy_column_name, df_sell, sell_column_name),net_volume_lot)
        net_ratio = round(net_ratio, 2)

        if(calculateNetValue(df_buy,"NBVal") > 500000000):
            #Append net ratio
            netRatioArray.append(net_ratio)

            #Change ticker string
            tickerCopy.append(x[:4])
        
        #LoadingBar
        count += 1
        if(upload == "no"):
            if(count == 1): print("\n")
            progress(count, len(tickers), status='  Calculating...')

    #Create table
    dictTickerRatio = {'Ticker':tickerCopy,'Ratio':netRatioArray}
    df_recommend = pd.DataFrame(dictTickerRatio)

    #Sort
    df_recommend.sort_values(by="Ratio", ascending=False, inplace = True)
    df_recommend.reset_index(drop=True, inplace=True)
    df_recommend.index += 1

    #Colors
    counter = 0
    for row in df_recommend.itertuples(index=True, name='Pandas'):
        x = float(getattr(row, "Ratio"))
        if(x >= 0):
            split_index = counter
        counter += 1

    #Split into two
    split_index += 1
    df1 = df_recommend.iloc[:split_index, :]
    df2 = df_recommend.iloc[split_index:, :]

    df1.columns = ['A_Ticker', 'A_Ratio']
    df2.columns = ['D_Ticker', 'D_Ratio']

    df2 = df2.sort_values(by="D_Ratio", ascending = True)

    df1.reset_index(drop=True, inplace=True)
    df2.reset_index(drop=True, inplace=True)

    #Combine dataframe
    final_result_dataframe = pd.concat([df1,df2], ignore_index = False, axis=1)
    final_result_dataframe.index += 1

    final_result_dataframe.fillna('-', inplace=True)

    #Colors
    for row in final_result_dataframe.itertuples(index=True, name='Pandas'):
        x = getattr(row, "A_Ratio")
        y = getattr(row, "D_Ratio")

        if(x != '-' and type(x) != str):
            if(x < 0):
                final_result_dataframe.replace(to_replace = x , value = str(fg.red + str(x) + fg.rs), inplace = True)
            else:
                final_result_dataframe.replace(to_replace = x , value = str(fg.green + str(x) + fg.rs), inplace = True)
    
        if(y != '-' and type(y) != str):
            if(y < 0):
                final_result_dataframe.replace(to_replace = y , value = str(fg.red + str(y) + fg.rs), inplace = True)
            else:
                final_result_dataframe.replace(to_replace = y , value = str(fg.green + str(y) + fg.rs), inplace = True)
    
    #Get date
    date1 = pd.read_excel(path + '\\daily_data\\' + date + '\\' + tempTicker, 'Sheet1', index_col=None, usecols = "D", header = 0, nrows=0)
    date1 = date1.columns.values[0]
    date2 = pd.read_excel(path + '\\daily_data\\' + date + '\\' + tempTicker, 'Sheet1', index_col=None, usecols = "F", header = 0, nrows=0)
    date2 = date2.columns.values[0]

    #Print
    os.system('cls')
    print(fg.yellow + "\nACCUM/DIST (Top " + str(n) + " net ratio) [" + str(date1[0:10]) + "]:" + fg.rs + "\n")
    print(tabulate(final_result_dataframe, headers='keys', tablefmt='fancy_grid', stralign="center", disable_numparse=True))
    print("\n\n")

def viewHistory(ticker, n):
    dateListTemp = os.listdir(path + '\daily_data/')
    ticker = ticker.upper()

    #List folder that contains ticker
    dateList = []
    i = 0
    while(i < len(dateListTemp)):
        if(os.path.isfile(path + '\daily_data\\' + dateListTemp[i] + "\\" + ticker + "ToBroker.xls")):
            dateList.append(dateListTemp[i])
        elif(os.path.isfile(path + '\daily_data\\' + dateListTemp[i] + "\\" + ticker + ".xlsx")):
            dateList.append(dateListTemp[i])
        i = i + 1

    #Loop all date
    totalLot = 0
    totalRatio = 0
    ratio = []
    netLotArray = []
    ypLoc = []
    pdLoc = []
    ccLoc = []
    kkLoc = []
    niLoc = []
    change = []
    count = 0
    for x in dateList:
        #Pecah file excel jadi dua (BUY & SELL)
        if(os.path.isfile(path + '\daily_data\\' + x + "\\" + ticker + "ToBroker.xls")):
            df_buy = newDataFrame(path + '\daily_data\\' + x + "\\" + ticker + "ToBroker.xls", 2, [0,1,2])
            df_sell = newDataFrame(path + '\daily_data\\' + x + "\\" + ticker + "ToBroker.xls", 2, [5,6,7])
        elif(os.path.isfile(path + '\daily_data\\' + x + "\\" + ticker + ".xlsx")):
            df_buy = newDataFrame(path + '\daily_data\\' + x + "\\" + ticker + ".xlsx", 2, [0,1,2])
            df_sell = newDataFrame(path + '\daily_data\\' + x + "\\" + ticker + ".xlsx", 2, [5,6,7])

        #nama column yang kita mau (sementara maybe(?))
        buy_column_name = list(df_buy.columns)[1] 
        sell_column_name = list(df_sell.columns)[1]

        #Reformat column dan ubah datatype ke float
        df_buy = reformatExcel(df_buy, buy_column_name)
        df_sell = reformatExcel(df_sell,sell_column_name)

        #Dipakai untuk calculate net ratio
        net_volume_lot = calculateNetVolumeLot(df_buy, buy_column_name)
        net_ratio = calculateNetRatio(calculateNetLot(n, df_buy, buy_column_name, df_sell, sell_column_name),net_volume_lot)
        net_ratio = round(net_ratio, 2)
        totalRatio = totalRatio + net_ratio

        #Find broker location
        YP_Location = findBroker(df_buy, df_sell, "YP")
        PD_Location = findBroker(df_buy, df_sell, "PD")
        CC_Location = findBroker(df_buy, df_sell, "CC")
        KK_Location = findBroker(df_buy, df_sell, "KK")
        NI_Location = findBroker(df_buy, df_sell, "NI")

        #Save ratio to array
        ratio.append(net_ratio)

        #Save net lot to array
        net_lot = int(calculateNetLot(n, df_buy, buy_column_name, df_sell, sell_column_name))
        totalLot = totalLot + net_lot
        netLotArray.append(str('{:,}'.format(int(net_lot))))

        #Save YP location to array
        ypLoc.append(YP_Location)
        pdLoc.append(PD_Location)
        ccLoc.append(CC_Location)
        kkLoc.append(KK_Location)
        niLoc.append(NI_Location)

        #Save price change to array
        try:
            tickerData = yf.Ticker(str(ticker + ".JK"))

            date = datetime.datetime.strptime(x, "%d-%m-%Y")
            one_day = datetime.timedelta(days=1) #For calculating date

            if(date.strftime("%A") == 'Monday'):
                date_before = date - (one_day*3)
            else:
                date_before = date - one_day

            date_after = date + one_day

            while True:
                tickerDf = tickerData.history(period='1d', start=date_before.strftime("%Y-%m-%d"), end=date_after.strftime("%Y-%m-%d"), debug = False)
                if(len(tickerDf) == 0):
                    raise Exception("Ticker delisted")
                if(len(tickerDf) > 1):
                    break
                date_before = date_before - one_day

            priceChange = tickerDf["Close"].iloc[1] - tickerDf["Close"].iloc[0]
            percentage = (priceChange * 100)/tickerDf["Close"].iloc[0]

            if(percentage < 0):
                percentage = fg.red + str('{:.2f}'.format(percentage)) + fg.rs
            else:
                percentage = fg.green + ' ' + str('{:.2f}'.format(percentage)) + fg.rs

            change.append(percentage)
        except:
            change.append("not found")
            pass

        #Progress bar
        count = count + 1
        if(count == 1): print('\n')
        progress(count, len(dateList), status='  Calculating ' + x + '...')
            
    #Status array
    statusArray = []
    for x in ratio:
        statusArray.append(displayNetRatioStatus(x))
    
    #Transform data
    if(len(change) != len(dateList)):
        dictDatesRatio = {'Date':dateList,'Ratio':ratio, 'Status':statusArray, 'Net_Lot':netLotArray, 'YP':ypLoc, 'PD':pdLoc, 'KK':kkLoc, 'CC':ccLoc, 'NI':niLoc}
    else:
        dictDatesRatio = {'Date':dateList,'Ratio':ratio, 'Status':statusArray, 'Net_Lot':netLotArray, 'Change':change, 'YP':ypLoc, 'PD':pdLoc, 'KK':kkLoc, 'CC':ccLoc, 'NI':niLoc}
    
    # dictDatesRatio = {'Date':dateList,'Ratio':ratio, 'Status':statusArray, 'Net_Lot':netLotArray, 'YP':ypLoc, 'PD':pdLoc, 'KK':kkLoc, 'CC':ccLoc, 'NI':niLoc}

    df_date_ratio = pd.DataFrame(dictDatesRatio)

    #Convert to date
    df_date_ratio['Date'] = pd.to_datetime(df_date_ratio['Date'], format='%d-%m-%Y')
    df_date_ratio['Date'] = df_date_ratio['Date'].dt.date
    
    #Sort and remove index
    df_date_ratio.sort_values(by="Date", ascending=False, inplace = True)
    df_date_ratio.reset_index(drop=True, inplace=True)
    df_date_ratio.index += 1

    #Colors
    for row in df_date_ratio.itertuples(index=True, name='Pandas'):
        x = float(getattr(row, "Ratio"))
        if(x < 0):
            df_date_ratio.replace(to_replace = x , value = str(fg.red + str(x) + fg.rs), inplace = True)
        else:
            df_date_ratio.replace(to_replace = x , value = str(fg.green + " " + str(x) + fg.rs), inplace = True)

    #Calculate avg ratio
    if(len(dateList) != 0):
        avgRatio = float(totalRatio/len(dateList))
    else:
        avgRatio = float(0)

    #Print table
    os.system('cls')
    print(fg.yellow + "\n" + ticker.upper() + " (Top " + str(n) + " net ratio) :" + fg.rs + "\n")
    print(tabulate(df_date_ratio, headers='keys', tablefmt='fancy_grid', stralign="center", showindex=False, disable_numparse=True))
    print("\nSUM Net Lot : " + str('{:,}'.format(int(totalLot))))
    print("\nRatio average : " + str("%.2f" % avgRatio))
    print('\n')

    #Print graph
    # net_lot_array = df_date_ratio['Net_Lot'].tolist()
    # testPlot(net_lot_array)

def viewDataOnDate(ticker, date):
    dateListTemp = os.listdir(path + '\daily_data/')
    ticker = ticker.upper()
    
    #List folder that contains ticker
    dateList = []
    i = 0
    while(i < len(dateListTemp)):
        if(os.path.isfile(path + '\daily_data\\' + dateListTemp[i] + "\\" + ticker + "ToBroker.xls")):
            dateList.append(dateListTemp[i])
        elif(os.path.isfile(path + '\daily_data\\' + dateListTemp[i] + "\\" + ticker + ".xlsx")):
            dateList.append(dateListTemp[i])
        i = i + 1

    if(date not in dateList):
        os.system('cls')
        print("No " + ticker + " data found on " + date + "\n")
        return
    else:
        #Pecah file excel jadi dua (BUY & SELL)
        if(os.path.isfile(path + '\daily_data\\' + date + "\\" + ticker + "ToBroker.xls")):
            df_buy = newDataFrame(path + '\daily_data\\' + date + "\\" + ticker + "ToBroker.xls", 2, [0,1,2])
            df_sell = newDataFrame(path + '\daily_data\\' + date + "\\" + ticker + "ToBroker.xls", 2, [5,6,7])
            df_view = newDataFrame(path + '\daily_data\\' + date + "\\" + ticker + "ToBroker.xls", 2, [0,1,2,3,5,6,7,8])
        elif(os.path.isfile(path + '\daily_data\\' + date + "\\" + ticker + ".xlsx")):
            df_buy = newDataFrame(path + '\daily_data\\' + date + "\\" + ticker + ".xlsx", 2, [0,1,2])
            df_sell = newDataFrame(path + '\daily_data\\' + date + "\\" + ticker + ".xlsx", 2, [5,6,7])
            df_view = newDataFrame(path + '\daily_data\\' + date + "\\" + ticker + ".xlsx", 2, [0,1,2,3,5,6,7,8])

        #nama column yang kita mau (sementara maybe(?))
        buy_column_name = list(df_buy.columns)[1] 
        sell_column_name = list(df_sell.columns)[1]

        #Reformat column dan ubah datatype ke float
        df_buy = reformatExcel(df_buy, buy_column_name)
        df_sell = reformatExcel(df_sell,sell_column_name)

        #Calculate net volume
        net_volume_lot = calculateNetVolumeLot(df_buy, buy_column_name)

        #Print result
        os.system('cls')
        print ("\n" + fg.yellow + ticker + " [" + date + "]" + fg.rs + "\n")
        
        #Remove NaN
        df_view.fillna("-", inplace=True)
        
        #Colour Retail broker
        df_view.replace(to_replace ="YP", value = str(fg.red + "YP" + fg.rs), inplace = True)
        df_view.replace(to_replace ="PD", value = str(fg.red + "PD" + fg.rs), inplace = True)
        df_view.replace(to_replace ="CC", value = str(fg.red + "CC" + fg.rs), inplace = True)
        df_view.replace(to_replace ="KK", value = str(fg.red + "KK" + fg.rs), inplace = True)
        df_view.replace(to_replace ="NI", value = str(fg.red + "NI" + fg.rs), inplace = True)
        
        #Print
        df_view.columns = [fg.green + 'Buyer' + fg.rs, fg.green + 'NBLot' + fg.rs, fg.green + 'NBVal' + fg.rs, fg.green + 'BAvg' + fg.rs,
                        fg.red + 'Seller' + fg.rs, fg.red + 'NSLot' + fg.rs, fg.red + 'NSVal' + fg.rs, fg.red + 'SAvg' + fg.rs]
        df_view.index = df_view.index + 1
        df_view.insert(4, str(fg.magenta + "No" + fg.rs), prettyIndex(df_view.index), True)
        print(tabulate(df_view, headers='keys', tablefmt='fancy_grid', stralign="center", showindex=False, disable_numparse=True))
        
        print("\n")

        #Print NET VOLUME
        print("NET VOLUME(LOT): " + str('{:,}'.format(int(net_volume_lot))) + "\n\n")

        #Print NET LOT
        print ("NET (LOT):")
        for x in range (5):
            top_status = calculateTop( x+1, df_buy, buy_column_name, df_sell, sell_column_name)
            net_lot = calculateNetLot(x+1, df_buy, buy_column_name, df_sell, sell_column_name)
            print(" - TOP "+ str(x+1) + ": " + "%-20s" % str(top_status) + " | Net Lot: " + str('{:,}'.format(int(net_lot))) )

        #Print NET RATIO
        print ("\nNET RATIO:")
        for x in range (5):
            net_ratio = calculateNetRatio(calculateNetLot(x+1, df_buy, buy_column_name, df_sell, sell_column_name),net_volume_lot)
            print(" - TOP "+ str(x+1) + ": " + displayNetRatioStatus(net_ratio).center(28)  + " | Net Ratio: " + str("%.2f" % net_ratio) )
        print("\n")
        return

def ipotScrape(start_dt, end_dt):
    path = getPath()

    #Progress counter
    progressCount = 0

    #Read Tickers (early for progress bar purpose)
    tickerList = getTickerList()

    ######################################################### DATE ###########################################################

    start_date = datetime.datetime.strptime(start_dt, "%d-%m-%Y")
    start_dt = start_date.strftime('%Y%m%d')
    end_date = datetime.datetime.strptime(end_dt, "%d-%m-%Y")
    end_dt = end_date.strftime('%Y%m%d')

    ###################################################### USER AGENT ########################################################

    user_agent_file = open(str(path + '\\bot_data\\UA.txt'), 'r')
    user_agent_list = user_agent_file.read().splitlines()

    user_agent = user_agent_list[random.randrange(0, len(user_agent_list))]

    #Headless and incog
    options = webdriver.ChromeOptions()
    options.add_argument('--ignore-certificate-errors')
    options.add_argument('--incognito')
    options.add_argument('--headless')
    options.add_argument('--log-level=3')
    options.add_experimental_option('excludeSwitches', ['enable-logging'])
    options.add_argument("user-agent=" + user_agent)

    ####################################################### OPEN TOR #########################################################

    while True:
        #Start TOR
        tor_path = str(path + '\\bot_data\\Tor\\Browser\\TorBrowser\\Tor\\tor.exe')
        tor_process = subprocess.Popen(tor_path)
        subprocess_list.append(tor_process)

        #Use TOR Proxy
        proxy = "socks5://localhost:9050"
        options.add_argument('--proxy-server=%s' % proxy)

        try:
            #Display IP Address Used (Optional)
            chrome = webdriver.Chrome(str(path + '\\bot_data\\chromedriver.exe'), options = options)
            chrome.get("https://wtfismyip.com/text")
            check_ip_html = chrome.page_source
            soup = BeautifulSoup(check_ip_html, 'lxml')
            if(soup.find("pre") is not None):
                print("\n")
                print('Downloading ' + str(len(tickerList)) + ' data using IP: ' + str(soup.find("pre").text))
                chrome.quit()
                break
        except Exception as e:
            print(e)
            pass

        tor_process.kill()
        input('\nConnection failed, press enter to retry...')
        print('\033[F                                          ')
    
    start = time.time()

    #################################################### OPENING IPOT ########################################################

    first_driver_flag = 0 #0 for not opened yet/correctly
    second_driver_flag = 0

    while True:
        if(first_driver_flag == 0):
            driver = webdriver.Chrome(str(path + '\\bot_data\\chromedriver.exe'), options = options)
        if(second_driver_flag == 0):
            driver2 = webdriver.Chrome(str(path + '\\bot_data\\chromedriver.exe'), options = options)

        if(first_driver_flag == 0):
            driver_list.append(driver)
        if(second_driver_flag == 0):
            driver_list.append(driver)
        
        progressCount = progressCount + 1
        progress(progressCount, (len(tickerList)//2)+3, status='  Opening page...')

        try:
            if(first_driver_flag == 0):
                driver.get("https://app12.ipotindonesia.com/ipot/tablet/ulogin.jsp")
            if(second_driver_flag == 0):
                driver2.get("https://app12.ipotindonesia.com/ipot/tablet/ulogin.jsp")
        except:
            if(first_driver_flag == 0):
                driver.quit()
            if(second_driver_flag == 0):
                driver2.quit()
            continue

        #Set wait time
        wait = WebDriverWait(driver, 20)
        wait2 = WebDriverWait(driver2, 20)

        try:
            #Click free to enter button
            if(first_driver_flag == 0):
                enter_button = wait.until(ec.visibility_of_element_located((By.XPATH, "//*[@id='hero']/ul/li/input")))
                enter_button.click()
            if(second_driver_flag == 0):
                enter_button2 = wait2.until(ec.visibility_of_element_located((By.XPATH, "//*[@id='hero']/ul/li/input")))
                enter_button2.click()

            #Enter full menu mode
            if(first_driver_flag == 0):
                full_menu_button = wait.until(ec.visibility_of_element_located((By.XPATH, "//*[@id='toppanel']/div/ul/li[2]/a/span")))
                full_menu_button.click()
            if(second_driver_flag == 0):
                full_menu_button2 = wait2.until(ec.visibility_of_element_located((By.XPATH, "//*[@id='toppanel']/div/ul/li[2]/a/span")))
                full_menu_button2.click()

            #Wait page to load
            time.sleep(10)
            progressCount = progressCount + 1
            progress(progressCount, (len(tickerList)//2)+3, status='  Loading page...')

            #Hover over securities analysis
            if(first_driver_flag == 0):
                sec_analysis_button = driver.find_elements_by_xpath("//*[@id='toppanel']/div/ul/li[8]")
                action = ActionChains(driver)
                action.move_to_element(sec_analysis_button[1]).perform()
            if(second_driver_flag == 0):
                sec_analysis_button2 = driver2.find_elements_by_xpath("//*[@id='toppanel']/div/ul/li[8]")
                action2 = ActionChains(driver2)
                action2.move_to_element(sec_analysis_button2[1]).perform()

            #Click to display broker summary
            if(first_driver_flag == 0):
                brok_sum_button = wait.until(ec.visibility_of_element_located((By.XPATH, "//*[@id='toppanel']/div/ul/li[8]/ul/li[3]/a")))
                brok_sum_button.click()
            if(second_driver_flag == 0):
                brok_sum_button2 = wait2.until(ec.visibility_of_element_located((By.XPATH, "//*[@id='toppanel']/div/ul/li[8]/ul/li[3]/a")))
                brok_sum_button2.click()

            #Enter ticker
            if(first_driver_flag == 0):
                ticker_input = wait.until(ec.visibility_of_element_located((By.XPATH, "//*[@id='pop_brokersum_stock']")))
                ticker_input.send_keys(u'\ue009' + u'\ue003')
                ticker_input.send_keys("TLKM")
            if(second_driver_flag == 0):
                ticker_input2 = wait2.until(ec.visibility_of_element_located((By.XPATH, "//*[@id='pop_brokersum_stock']")))
                ticker_input2.send_keys(u'\ue009' + u'\ue003')
                ticker_input2.send_keys("TLKM")

            #Enter dates
            if(first_driver_flag == 0):
                date_from = wait.until(ec.visibility_of_element_located((By.XPATH, "//*[@id='pop_bas_from']")))
                date_from.send_keys(u'\ue009' + u'\ue003')
                date_from.send_keys(start_dt)
                date_to = wait.until(ec.visibility_of_element_located((By.XPATH, "//*[@id='pop_bas_to']")))
                date_to.send_keys(u'\ue009' + u'\ue003')
                date_to.send_keys(start_dt)

            if(second_driver_flag == 0):
                date_from2 = wait2.until(ec.visibility_of_element_located((By.XPATH, "//*[@id='pop_bas_from']")))
                date_from2.send_keys(u'\ue009' + u'\ue003')
                date_from2.send_keys(start_dt)
                date_to2 = wait2.until(ec.visibility_of_element_located((By.XPATH, "//*[@id='pop_bas_to']")))
                date_to2.send_keys(u'\ue009' + u'\ue003')
                date_to2.send_keys(start_dt)

            #CLick show
            if(first_driver_flag == 0):
                show_button = wait.until(ec.visibility_of_element_located((By.XPATH, "//*[@id='popupBrokerSum']/div/table[1]/tbody/tr[2]/td[4]/input")))
                show_button.click()
            if(second_driver_flag == 0):
                show_button2 = wait2.until(ec.visibility_of_element_located((By.XPATH, "//*[@id='popupBrokerSum']/div/table[1]/tbody/tr[2]/td[4]/input")))
                show_button2.click()

        except:
            progressCount = progressCount - 1
            progress(progressCount, (len(tickerList)//2)+3, status='  Retrying......')
            if(first_driver_flag == 0):
                driver.quit()
            if(second_driver_flag == 0):
                driver2.quit()
            continue

        time.sleep(5)

        #Get initial data to check
        page_source = driver.page_source
        page_source2 = driver2.page_source

        #Prepare
        html_soup = BeautifulSoup(page_source, 'lxml')

        table = html_soup.find(id = 'pop_tblBrokerSum')
        x = []
        array = []

        for row in table.findAll('tr'):
            count_row = 1
            count = 0
            columns = row.findAll('td')
            for column in columns:
                x.append(column.get_text())
                count = count + 1
                if(count == 10):
                    array.append(x)
                    x = []
            count_row = count_row + 1

        test_df = pd.DataFrame(array, columns = ["BUY", "BLot", "BVal", "BAvg", "?", "NO", "SL", "SLot", "SVal", "SAvg"])
        test_df.drop('?', axis =1, inplace = True)

        #Prepare
        html_soup2 = BeautifulSoup(page_source2, 'lxml')

        table2 = html_soup2.find(id = 'pop_tblBrokerSum')
        x2 = []
        array2 = []

        for row in table2.findAll('tr'):
            count_row = 1
            count = 0
            columns = row.findAll('td')
            for column in columns:
                x2.append(column.get_text())
                count = count + 1
                if(count == 10):
                    array2.append(x2)
                    x2 = []
            count_row = count_row + 1

        test_df2 = pd.DataFrame(array2, columns = ["BUY", "BLot", "BVal", "BAvg", "?", "NO", "SL", "SLot", "SVal", "SAvg"])
        test_df2.drop('?', axis =1, inplace = True)

        if(test_df.empty == False):
            first_driver_flag = 1

        if(test_df2.empty == False):
            second_driver_flag = 1

        if(test_df.empty == False and test_df2.empty == False):
            break

        if(first_driver_flag == 0):
            driver.quit()
        if(second_driver_flag == 0):
            driver2.quit()
        progressCount = progressCount - 2
        progress(progressCount, (len(tickerList)//2)+3, status='  Retrying......')

    ################################################## START OF LOOP #########################################################

    make_dir_flag = 0

    #Split list
    half = len(tickerList)//2
    first_half_ticker_list = tickerList[:half]
    second_half_ticker_list = tickerList[half:]

    max_length = findMax(len(first_half_ticker_list), len(second_half_ticker_list))
    min_length = findMin(len(first_half_ticker_list), len(second_half_ticker_list))

    for i in range(min_length):
        #Randomize interval
        random_var = random.uniform(6.0, 7.0)
        
        #Enter ticker
        ticker_input = wait.until(ec.visibility_of_element_located((By.XPATH, "//*[@id='pop_brokersum_stock']")))
        ticker_input.send_keys(u'\ue009' + u'\ue003')
        ticker_input.send_keys(first_half_ticker_list[i])
        ticker_input2 = wait2.until(ec.visibility_of_element_located((By.XPATH, "//*[@id='pop_brokersum_stock']")))
        ticker_input2.send_keys(u'\ue009' + u'\ue003')
        ticker_input2.send_keys(second_half_ticker_list[i])

        #Enter dates
        date_from = wait.until(ec.visibility_of_element_located((By.XPATH, "//*[@id='pop_bas_from']")))
        date_from.send_keys(u'\ue009' + u'\ue003')
        date_from.send_keys(start_dt)
        date_from2 = wait2.until(ec.visibility_of_element_located((By.XPATH, "//*[@id='pop_bas_from']")))
        date_from2.send_keys(u'\ue009' + u'\ue003')
        date_from2.send_keys(start_dt)

        date_to = wait.until(ec.visibility_of_element_located((By.XPATH, "//*[@id='pop_bas_to']")))
        date_to.send_keys(u'\ue009' + u'\ue003')
        date_to.send_keys(end_dt)
        date_to2 = wait2.until(ec.visibility_of_element_located((By.XPATH, "//*[@id='pop_bas_to']")))
        date_to2.send_keys(u'\ue009' + u'\ue003')
        date_to2.send_keys(end_dt)

        #Change to net
        drow_down_button = wait.until(ec.visibility_of_element_located((By.XPATH, "//*[@id='pop_basMode']/option[2]")))
        drow_down_button.click()
        drow_down_button2 = wait2.until(ec.visibility_of_element_located((By.XPATH, "//*[@id='pop_basMode']/option[2]")))
        drow_down_button2.click()

        #Change to all trade
        drow_down_button = wait.until(ec.visibility_of_element_located((By.XPATH, "//*[@id='pop_basBoard']/option[1]")))
        drow_down_button.click()
        drow_down_button2 = wait2.until(ec.visibility_of_element_located((By.XPATH, "//*[@id='pop_basBoard']/option[1]")))
        drow_down_button2.click()

        #CLick show
        show_button = wait.until(ec.visibility_of_element_located((By.XPATH, "//*[@id='popupBrokerSum']/div/table[1]/tbody/tr[2]/td[4]/input")))
        show_button.click()
        show_button2 = wait2.until(ec.visibility_of_element_located((By.XPATH, "//*[@id='popupBrokerSum']/div/table[1]/tbody/tr[2]/td[4]/input")))
        show_button2.click()

        #Show progress
        progressCount = progressCount + 1
        progress(progressCount, (len(tickerList)//2)+3, status='  Downloading data......')

        #Wait for data to load
        time.sleep(random_var)
        # _ = wait.until(ec.visibility_of_element_located((By.XPATH, "//*[@id='pop_tblBrokerSum']")))

        page_source = driver.page_source
        page_source2 = driver2.page_source

        # Prepare
        html_soup = BeautifulSoup(page_source, 'lxml')

        table = html_soup.find(id = 'pop_tblBrokerSum')
        x = []
        array = []

        for row in table.findAll('tr'):
            count_row = 1
            count = 0
            columns = row.findAll('td')
            for column in columns:
                x.append(column.get_text())
                count = count + 1
                if(count == 10):
                    array.append(x)
                    x = []
            count_row = count_row + 1

        result_df = pd.DataFrame(array, columns = ["NBY", "NBLot", "NBVal", "BAvg", "?", "#", "NSL", "NSLot", "NSVal", "SAvg"])
        result_df.drop('?', axis =1, inplace = True)

        # Prepare
        html_soup2 = BeautifulSoup(page_source2, 'lxml')

        table2 = html_soup2.find(id = 'pop_tblBrokerSum')
        x2 = []
        array2 = []

        for row in table2.findAll('tr'):
            count_row = 1
            count = 0
            columns = row.findAll('td')
            for column in columns:
                x2.append(column.get_text())
                count = count + 1
                if(count == 10):
                    array2.append(x2)
                    x2 = []
            count_row = count_row + 1

        result_df2 = pd.DataFrame(array2, columns = ["NBY", "NBLot", "NBVal", "BAvg", "?", "#", "NSL", "NSLot", "NSVal", "SAvg"])
        result_df2.drop('?', axis =1, inplace = True)

        #Saving to excel
        if(len(result_df.index) > 1):
            #Make folder
            if(make_dir_flag == 0):
                path = os.getcwd()
                if not os.path.exists(path + '\\daily_data\\' + start_date.strftime('%d-%m-%Y') + '\\'):
                    os.makedirs(path + '\\daily_data\\' + start_date.strftime('%d-%m-%Y') + '\\')
                make_dir_flag = 1

            info_df = pd.DataFrame(columns=['Ticker', str(first_half_ticker_list[i]),'Start_dt', str(start_date.strftime('%Y-%m-%d')),'End_dt', str(end_date.strftime('%Y-%m-%d'))])
   
            #Write to Excel
            if(start_dt == end_dt):
                writer = pd.ExcelWriter(str(path + '\\daily_data\\' + start_date.strftime('%d-%m-%Y') + '\\' + str(first_half_ticker_list[i]) + '.xlsx'), engine='openpyxl')
            else:
                writer = pd.ExcelWriter(str(path + '\\weekly_data\\' + str(first_half_ticker_list[i]) + '.xlsx'), engine='openpyxl')
            info_df.to_excel(writer, index = False)
            result_df.to_excel(writer, index = False, startrow=2)
            writer.save()

        if(len(result_df2.index) > 1):
            #Make folder
            if(make_dir_flag == 0):
                path = os.getcwd()
                if not os.path.exists(path + '\\daily_data\\' + start_date.strftime('%d-%m-%Y') + '\\'):
                    os.makedirs(path + '\\daily_data\\' + start_date.strftime('%d-%m-%Y') + '\\')
                make_dir_flag = 1

            info_df2 = pd.DataFrame(columns=['Ticker', str(second_half_ticker_list[i]),'Start_dt', str(start_date.strftime('%Y-%m-%d')),'End_dt', str(end_date.strftime('%Y-%m-%d'))])

            #Write to Excel
            if(start_dt == end_dt):
                writer2 = pd.ExcelWriter(str(path + '\\daily_data\\' + start_date.strftime('%d-%m-%Y') + '\\' + str(second_half_ticker_list[i]) + '.xlsx'), engine='openpyxl')
            else:
                writer2 = pd.ExcelWriter(str(path + '\\weekly_data\\' + str(second_half_ticker_list[i]) + '.xlsx'), engine='openpyxl')
            info_df2.to_excel(writer2, index = False)
            result_df2.to_excel(writer2, index = False, startrow=2)
            writer2.save()

    if(min_length != max_length):
        #Randomize interval
        random_var = random.uniform(5.0, 7.0)
        
        #Enter ticker
        ticker_input2 = wait2.until(ec.visibility_of_element_located((By.XPATH, "//*[@id='pop_brokersum_stock']")))
        ticker_input2.send_keys(u'\ue009' + u'\ue003')
        ticker_input2.send_keys(second_half_ticker_list[max_length-1])

        #Enter dates
        date_from2 = wait2.until(ec.visibility_of_element_located((By.XPATH, "//*[@id='pop_bas_from']")))
        date_from2.send_keys(u'\ue009' + u'\ue003')
        date_from2.send_keys(start_dt)

        date_to2 = wait2.until(ec.visibility_of_element_located((By.XPATH, "//*[@id='pop_bas_to']")))
        date_to2.send_keys(u'\ue009' + u'\ue003')
        date_to2.send_keys(end_dt)

        #Change to net
        drow_down_button2 = wait2.until(ec.visibility_of_element_located((By.XPATH, "//*[@id='pop_basMode']/option[2]")))
        drow_down_button2.click()

        #Change to all trade
        drow_down_button2 = wait2.until(ec.visibility_of_element_located((By.XPATH, "//*[@id='pop_basBoard']/option[1]")))
        drow_down_button2.click()

        #CLick show
        show_button2 = wait2.until(ec.visibility_of_element_located((By.XPATH, "//*[@id='popupBrokerSum']/div/table[1]/tbody/tr[2]/td[4]/input")))
        show_button2.click()

        #Wait for data to load
        time.sleep(random_var)

        page_source2 = driver2.page_source

        # Prepare
        html_soup2 = BeautifulSoup(page_source2, 'lxml')

        table2 = html_soup2.find(id = 'pop_tblBrokerSum')
        x2 = []
        array2 = []

        for row in table2.findAll('tr'):
            count_row = 1
            count = 0
            columns = row.findAll('td')
            for column in columns:
                x2.append(column.get_text())
                count = count + 1
                if(count == 10):
                    array2.append(x2)
                    x2 = []
            count_row = count_row + 1

        result_df2 = pd.DataFrame(array2, columns = ["NBY", "NBLot", "NBVal", "BAvg", "?", "#", "NSL", "NSLot", "NSVal", "SAvg"])
        result_df2.drop('?', axis =1, inplace = True)

        #Saving to excel
        if(len(result_df2.index) > 1):
            #Make folder
            if(make_dir_flag == 0):
                path = os.getcwd()
                if not os.path.exists(path + '\\daily_data\\' + start_date.strftime('%d-%m-%Y') + '\\'):
                    os.makedirs(path + '\\daily_data\\' + start_date.strftime('%d-%m-%Y') + '\\')
                make_dir_flag = 1

            info_df2 = pd.DataFrame(columns=['Ticker', str(second_half_ticker_list[max_length-1]),'Start_dt', str(start_date.strftime('%Y-%m-%d')),'End_dt', str(end_date.strftime('%Y-%m-%d'))])

            #Write to Excel
            if(start_dt == end_dt):
                writer2 = pd.ExcelWriter(str(path + '\\daily_data\\' + start_date.strftime('%d-%m-%Y') + '\\' + str(second_half_ticker_list[max_length-1]) + '.xlsx'), engine='openpyxl')
            else:
                writer2 = pd.ExcelWriter(str(path + '\\weekly_data\\' + str(second_half_ticker_list[max_length-1]) + '.xlsx'), engine='openpyxl')
            info_df2.to_excel(writer2, index = False)
            result_df2.to_excel(writer2, index = False, startrow=2)
            writer2.save()

    #Close chrome
    driver.quit()
    driver2.quit()

    #Close TOR
    tor_process.kill()

    end = time.time()

    progressCount = progressCount + 1
    progress((len(tickerList)//2)+3, (len(tickerList)//2)+3, status="Process completed successfully (" + str(round((end-start)/60, 2)) + " Minutes)\n\n")

    input('Press enter to continue...')

    return

class tickerStd:
    ticker = ''
    n_std = 0.0
    date = ''

def checkSpike(ticker, number_of_day):

    curr_date = datetime.date.today()
    one_day = datetime.timedelta(days=1)

    start_dt = curr_date - (one_day * number_of_day)
    end_dt = curr_date + one_day

    tickerData = yf.Ticker(ticker + '.JK')

    count = 0
    while True:
        tickerDf = tickerData.history(period = '1d', start = start_dt.strftime('%Y-%m-%d'), end = end_dt.strftime('%Y-%m-%d'), debug = False)
        if(len(tickerDf) == 0):
            return
        if(len(tickerDf) >= number_of_day):
            break
        start_dt = start_dt - one_day
        if(count > number_of_day * 2):
            return
        count += 1

    tickerDf.reset_index(inplace=True)

    volume_list = tickerDf['Volume'].to_list()
    date_list = tickerDf['Date'].to_list()

    standard_dev = np.std(volume_list)
    mean = np.mean(volume_list)

    tickerStdClass = tickerStd()

    index = 0
    for volume in volume_list:
        difference = volume - mean
        if(standard_dev != 0):
            n_std = difference/standard_dev
        else:
            n_std = 0
        if(n_std >= 2):
            tickerStdClass.ticker = ticker
            tickerStdClass.n_std = round(n_std, 2)
            tickerStdClass.date = date_list[index]

            return tickerStdClass
        index += 1

def checkSpikeWrapper(x):
    return checkSpike(*x)

def scanSpike(number_of_day):
    tickerList = getSpikeTickerList()

    max_spike = len(tickerList)

    ticker_with_spike = []
    progress_count = 0

    args = ((ticker, number_of_day) for ticker in tickerList)

    result_ticker_list = []
    result_nstd_list = []
    result_date_list = []
    count = 0
    print('\n')
    progress(0, len(tickerList), 'Loading...')
    with concurrent.futures.ThreadPoolExecutor(max_workers = 100) as executor:
        # results = executor.map(checkSpikeWrapper, args)
        for result in executor.map(checkSpikeWrapper, args):
            # os.system('cls')
            progress(count, len(tickerList), 'Loading...')
            #check if the returned class is not empty
            if(result is not None):
                result_ticker_list.append(result.ticker)
                result_nstd_list.append(result.n_std)
                result_date_list.append(result.date)
            count += 1

    print('done')

    final_ticker_list = list(result_ticker_list)
    final_ticker_list = list(filter(None, final_ticker_list))

    final_nstd_list = list(result_nstd_list)
    final_nstd_list = list(filter(None, final_nstd_list))

    temp_date_list = list(result_date_list)
    final_date_list = []
    for date in temp_date_list:
        # temp_date = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
        final_date_list.append(date.strftime('%d-%m-%Y'))
    final_date_list = list(filter(None, final_date_list))

    result_df = pd.DataFrame(data = zip(final_ticker_list, final_nstd_list, final_date_list), columns = ['Ticker', 'N_Std', 'Date'])
    result_df.sort_values('N_Std', inplace = True, ascending = False)
    result_df.reset_index(drop = True, inplace = True)
    result_df.index += 1

    #OUTPUT
    os.system('cls')
    print(fg.yellow + '\nList of tickers with volume spike in last ' + str(number_of_day) + ' day(s):\n' + fg.rs)
    print(tabulate(result_df, headers='keys', tablefmt='fancy_grid', stralign="center", disable_numparse=True))
    print('\n')

def killAllSubprocess():
    timeout_sec = 5
    for p in subprocess_list:
        try:
            p.kill()
        except:
            pass

def killAllDriver():
    timeout_sec = 5
    for p in driver_list:
        try:
            p.quit()
        except:
            pass

def getTickerList():
    file = open('tickerList.txt', 'r')
    tickerList = file.read().splitlines()

    return tickerList

def getSpikeTickerList():
    file = open('SpikeTickerList.txt', 'r')
    tickerList = file.read().splitlines()

    return tickerList

def findMax(x, y):
    if(x >= y):
        return x
    elif(y > x):
        return y

def findMin(x, y):
    if(x <= y):
        return x
    elif(y < y):
        return y

################################################################ INITIAL DECLARATION/GLOBAL VAR ####################################################

path = getPath()
subprocess_list = []
driver_list = []
tickerList = os.listdir( path + '\\weekly_data/')
pd.options.display.float_format = "{:,.0f}".format
welcomeMessages()

########################################################################### MAIN ###################################################################
while(True):
    #input
    flag = 0
    exitFlag = 0
    while(flag == 0):
        print(fg.yellow + "Input command (\"/commands\" to view commands):" + fg.rs)
        ticker = input(">> ")
        # try:
        if(ticker == ""):
            welcomeMessages()
            flag = 0
        elif(ticker == "/commands"):
            welcomeMessages()
            flag = 0
        elif(ticker == "/download"):
            choice = input('>> Range or single [0/1]: ')
            if(int(choice) == 0):
                start_date = input(">> Start date (dd-mm-yyyy): ")
                end_date = input(">> End date (dd-mm-yyyy): ")
                ipotScrape(start_date, end_date)
            elif(int(choice) == 1):
                date = input(">> Input date (dd-mm-yyyy): ")
                ipotScrape(date, date)
                welcomeMessages()
                flag = 0
        elif(str(ticker).lower() == "view"):
            viewList()
            flag = 0
        elif(str(ticker[0:9]).lower() == "recommend" and len(ticker) > 10):
            recommendBest(tickerList, int(ticker[10:len(ticker)]))
            flag = 0
        elif(str(ticker[0:7]).lower() == "history" and len(ticker) > 13):
            viewHistory(str(ticker[8:12]), int(ticker[13:len(ticker)]))
            flag = 0
        elif(str(ticker[0:5]).lower() == "daily" and len(ticker) > 20):
            viewDataOnDate(str(ticker[6:10]), str(ticker[11:len(ticker)]))
            flag = 0
        elif(str(ticker[0:3]).lower() == "top" and len(ticker) > 10):
            topDaily(str(ticker[6:16]), int(ticker[4]))
            flag = 0
        elif(str(ticker[0:5]).lower() == "spike" and len(ticker) > 6):
            scanSpike(int(ticker[6:]))
            flag = 0
        elif(ticker == "/q"):
            exitFlag = 1
            break
        elif(str(ticker.upper() + 'ToBroker.xls') in tickerList or str(ticker.upper() + '.xlsx') in tickerList):
            flag = 1
            ticker = ticker.upper()
        else:
            os.system('cls')
            print("Ticker not found!\n")
            flag = 0
        # except:
        #     os.system('cls')
        #     print("\nUnknown command, please try again")
        #     flag = 0
            
    #Exit        
    if(exitFlag == 1):
        break
    
    os.system('cls')
    #Pecah file excel jadi dua (BUY & SELL)
    if(os.path.isfile(path + '\\weekly_data\\' + ticker + 'ToBroker.xls')):
        df_buy = newDataFrame(path + '\\weekly_data\\' + ticker + 'ToBroker.xls', 2, [0,1,2])
        df_sell = newDataFrame(path + '\\weekly_data\\' + ticker + 'ToBroker.xls', 2, [5,6,7])
        df_view = newDataFrame(path + '\\weekly_data\\' + ticker + 'ToBroker.xls', 2, [0,1,2,3,5,6,7,8])
    elif(os.path.isfile(path + '\weekly_data\\' + ticker + ".xlsx")):
        df_buy = newDataFrame(path + '\\weekly_data\\' + ticker + '.xlsx', 2, [0,1,2])
        df_sell = newDataFrame(path + '\\weekly_data\\' + ticker + '.xlsx', 2, [5,6,7])
        df_view = newDataFrame(path + '\\weekly_data\\' + ticker + '.xlsx', 2, [0,1,2,3,5,6,7,8])

    #nama column yang kita mau (sementara maybe(?))
    buy_column_name = list(df_buy.columns)[1] 
    sell_column_name = list(df_sell.columns)[1]

    #Reformat column dan ubah datatype ke float
    df_buy = reformatExcel(df_buy, buy_column_name)
    df_sell = reformatExcel(df_sell, sell_column_name)

    #Calculate net volume
    net_volume_lot = calculateNetVolumeLot(df_buy, buy_column_name)

    #Print result
    print ("\n" + fg.yellow + ticker + fg.rs + "\n")

    #Remove NaN
    df_view.fillna("-", inplace=True)
    
    #Colour Retail broker
    df_view.replace(to_replace ="YP", value = str(fg.red + "YP" + fg.rs), inplace = True)
    df_view.replace(to_replace ="PD", value = str(fg.red + "PD" + fg.rs), inplace = True)
    df_view.replace(to_replace ="CC", value = str(fg.red + "CC" + fg.rs), inplace = True)
    df_view.replace(to_replace ="KK", value = str(fg.red + "KK" + fg.rs), inplace = True)
    df_view.replace(to_replace ="NI", value = str(fg.red + "NI" + fg.rs), inplace = True)
    
    #Print
    df_view.columns = [fg.green + 'Buyer' + fg.rs, fg.green + 'NBLot' + fg.rs, fg.green + 'NBVal' + fg.rs, fg.green + 'BAvg' + fg.rs,
                       fg.red + 'Seller' + fg.rs, fg.red + 'NSLot' + fg.rs, fg.red + 'NSVal' + fg.rs, fg.red + 'SAvg' + fg.rs]
    df_view.index = df_view.index + 1
    df_view.insert(4, str(fg.magenta + "No" + fg.rs), prettyIndex(df_view.index), True)
    print(tabulate(df_view, headers='keys', tablefmt='fancy_grid', stralign="center", showindex=False, disable_numparse=True))
    
    print("\n")

    #Print NET VOLUME
    print("NET VOLUME(LOT): " + str('{:,}'.format(int(net_volume_lot))) + "\n\n")

    #Print NET LOT
    print ("NET (LOT):")
    for x in range (5):
        top_status = calculateTop( x+1, df_buy, buy_column_name, df_sell, sell_column_name)
        net_lot = calculateNetLot(x+1, df_buy, buy_column_name, df_sell, sell_column_name)
        print(" - TOP "+ str(x+1) + ": " + "%-20s" % str(top_status) + " | Net Lot: " + str('{:,}'.format(int(net_lot))) )

    #Print NET RATIO
    print ("\nNET RATIO:")
    for x in range (5):
        net_ratio = calculateNetRatio(calculateNetLot(x+1, df_buy, buy_column_name, df_sell, sell_column_name),net_volume_lot)
        print(" - TOP "+ str(x+1) + ": " + displayNetRatioStatus(net_ratio).center(28)  + " | Net Ratio: " + str("%.2f" % net_ratio) )
    print("\n")

###################################################################### RUN FUNC ON EXIT ############################################################

#Kill all subprocess on exit
atexit.register(killAllSubprocess)
atexit.register(killAllDriver)

# net volume lot 		: Jumlah semua lot satu sisi
# top n volume lot	: Selisih jumlah lot top n broker buyer dan seller

# net ratio 		: (net volume lot/top n volume lot) * 100