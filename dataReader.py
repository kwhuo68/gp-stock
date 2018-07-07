from datetime import datetime
import pandas as pd
import requests
import io
import re
import time

class DataReader():
    def __init__(self):
        self.cookie = None
        self.crumb = None
        self.fields = ['history']

    #Extract cookie and crumb from req
    def get_cookie_and_crumb(self):
        r = requests.get('https://finance.yahoo.com/quote/SPY')
        if r.cookies is None:
            raise Exception('No cookie found')
        else:
            self.cookie = r.cookies
        match = re.search('.*"CrumbStore":\{"crumb":"(?P<crumb>[^"]+)"\}', r.text)
        if match is None:
            raise Exception('No crumb found')
        else:
            self.crumb = match.group(1)

    #Get data for ticker
    def get_ticker(self, ticker, years_back = 5):
        if(self.cookie is None or self.crumb is None):
            self.get_cookie_and_crumb()

        start_date = datetime.today().replace(year=datetime.today().year - years_back)

        #Daiy prices, from years_back years ago until now
        params = {
            'period1': int(start_date.timestamp()),
            'period2': int(datetime.today().timestamp()),
            'events': 'history',
            'crumb': self.crumb,
            'interval': '1d'
        }

        url = 'https://query1.finance.yahoo.com/v7/finance/download/{}'.format(ticker)
        r = requests.get(url, params = params, cookies = self.cookie)

        df = pd.read_csv(io.BytesIO(r.content), sep = ',', error_bad_lines = False)
        df.set_index(pd.DatetimeIndex(df['Date']), inplace=True)
        df.drop('Date', axis=1, inplace=True)
        return df



