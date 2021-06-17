from requests import get
from bs4 import BeautifulSoup
import datetime
import sys
import ctypes

def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '#' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s %s\r' % (bar, percents, '%', status))
    sys.stdout.flush()

ctypes.windll.kernel32.SetConsoleTitleW("VerifyTickers")
file = open("TickerList.txt", "r")
tickersTemp = file.read().splitlines() 
file.close()

print("Progress :")
verifiedTickers = []
count = 0
for ticker in tickersTemp:
    progress(count, len(tickersTemp), status='')
    count += 1
    #Check ticker exist
    currentDate = datetime.datetime.now()
    dateTest = str(currentDate.month) + "/" + str(currentDate.day) + "/" + str(currentDate.year)
    testLink = "https://www.indopremier.com/module/saham/include/data-brokersummary.php?code=" + ticker + "&start=" + dateTest + "&end=" + dateTest + "&fd=all&board=all"
    response = get(testLink)
    html_soup = BeautifulSoup(response.text, 'html.parser')
    if (html_soup.div is not None):
        verifiedTickers.append(ticker)

file = open("TickerList.txt", "w")
file.write('\n'.join(verifiedTickers))
file.close()
