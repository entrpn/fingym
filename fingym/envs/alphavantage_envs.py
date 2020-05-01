import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np
import time

from fingym.envs.env import Env

from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators

data_folder = os.path.join(os.path.dirname(__file__),'..','data/')

class AlphavantageDailyEnv(Env):
  def __init__(self, symbol, alphavantage_key, update_data = False):
    self.symbol = symbol
    self.alphavantage_key = alphavantage_key
    self.date_str_format = '%Y-%m-%d'
    self.update_data = update_data
    self.data = self._load_data()
    self.n_step = len(self.data)
    self.cur_step = 0
    self.initial_investment = 25000
    self.cash_in_hand = self.initial_investment
    self._close_idx = 4
    self._open_idx = 1
    self.stock_owned = 0

    # stock_owned, cash_in_hand, date, hclo, volume
    self.state_dim = 2 + self.data.shape[1]

    # action[0]
    # 0 - hold/do nothing
    # 1 - buy
    # 2 - hold
    # action[1]
    # Number of actions to buy/sell
    self.action_size = 2

  def reset(self):
    self.cur_step = 0
    self.cash_in_hand = self.initial_investment
    self.stock_owned = 0
    return self._get_obs() 

  def step(self, action):
    
    prev_val = self._get_val()
    if self.cur_step < self.n_step -1:
        self.cur_step += 1

    self._trade(action)

    # get the new value after taking the action
    cur_val = self._get_val()

    reward = cur_val - prev_val
    obs = self._get_obs()
    done = self.cur_step >= self.n_step - 1
    info = {'cur_val': cur_val}

    return obs, reward, done, info

  def _trade(self, action):
    
    action[1] = round(action[1])
    # Mimic buying/selling the next day at open
    # from getting the previous day's prices.
    stock_price = self._get_stock_price_open()

    # sell
    shares_to_sell = 0
    if action[0] == 2:
        shares_to_sell = action[1]
    
    # buy
    shares_to_buy = 0
    if action[0] ==1:
        shares_to_buy = action[1]

    if shares_to_sell > 0:
        # sell all shares
        if self.stock_owned < shares_to_sell:
            self.cash_in_hand += self.stock_owned * stock_price
            self.stock_owned = 0
        else:
            self.stock_owned -= shares_to_sell
            self.cash_in_hand += shares_to_sell * stock_price
    
    
    if shares_to_buy > 0:
        can_buy = True
        shares_bought = 0
        while can_buy:
            if self.cash_in_hand > stock_price and shares_bought < shares_to_buy:
                self.stock_owned += 1
                shares_bought += 1
                self.cash_in_hand -= stock_price
            else:
                can_buy = False

  def _get_obs(self):
    """
    Return array with number stock owned, stock price and cash in hand.
    Ex: [stock_owned, cash_in_hand, date, hclo, volume]
    """
    obs = np.empty(self.state_dim)
    obs[0] = self.stock_owned
    obs[1] = self.cash_in_hand
    # date
    obs[2] = self.data[self.cur_step][0]
    obs[3:] = self.data[self.cur_step][1:]
    return obs

  def _get_val(self):
    return self.stock_owned * self._get_stock_price_close() + self.cash_in_hand

  def _get_stock_price_close(self):
    return self.data[self.cur_step][self._close_idx]

  def _get_stock_price_open(self):
    return self.data[self.cur_step][self._open_idx]

  def _load_data(self):
    
    csv_path = data_folder+f'{self.symbol}_daily.csv'

    daily_path_exists = os.path.exists(csv_path)

    if not daily_path_exists or self.update_data:
      service = AlphaVantageService(self.symbol, self.alphavantage_key)
      service.prepareStockData()

    ohclv_data = pd.read_csv(csv_path).to_numpy()
  
    # get sma 9, 20, 50, 200
    csv_path = data_folder+f'{self.symbol}_sma_9.csv'
    sma_9 = pd.read_csv(csv_path).to_numpy()

    csv_path = data_folder+f'{self.symbol}_sma_50.csv'
    sma_50 = pd.read_csv(csv_path).to_numpy()

    csv_path = data_folder+f'{self.symbol}_sma_200.csv'
    sma_200 = pd.read_csv(csv_path).to_numpy()
    
    # get bbands, this gives me the 20 sma as well
    csv_path = data_folder+f'{self.symbol}_bb_20.csv'
    bb_20 = pd.read_csv(csv_path).to_numpy()
    
    # get rsi
    csv_path = data_folder+f'{self.symbol}_rsi.csv'
    rsi = pd.read_csv(csv_path).to_numpy()

    # get cci
    csv_path = data_folder+f'{self.symbol}_cci.csv'
    cci = pd.read_csv(csv_path).to_numpy()
    
    # get obv
    csv_path = data_folder+f'{self.symbol}_obv.csv'
    obv = pd.read_csv(csv_path).to_numpy()

    data = np.zeros((len(sma_200),15))

    assert ohclv_data.shape[1] == 6
    ohclv_data = {a : (b,c,d,e,f) for a,b,c,d,e,f in ohclv_data}
    sma_9 = {a : b for a,b in sma_9}
    sma_50 = {a : b for a,b in sma_50}
    sma_200 = {a : b for a,b in sma_200}
    bb_20 = {a: (b,c,d) for a,b,c,d in bb_20}
    rsi = {a: b for a,b in rsi}
    cci = {a: b for a,b in cci}
    obv = {a: b for a,b in obv}

    i=0
    for key in sma_200:    
      data[i][0] = self._convert_date_str_to_datetime(key)
      data[i][1:6] = ohclv_data.get(key)
      data[i][6] = sma_9.get(key)
      data[i][7] = bb_20.get(key)[-1]
      data[i][8] = sma_50.get(key)
      data[i][9] = sma_200.get(key)
      data[i][10] = rsi.get(key)
      data[i][11] = cci.get(key)
      data[i][12] = obv.get(key)
      data[i][13] = bb_20.get(key)[0]
      data[i][14] = bb_20.get(key)[1]
      i+=1

    return data

  def _convert_date_str_to_datetime(self,datestr):
    epoch = datetime.utcfromtimestamp(0)
    if sys.version_info[1] <=5:
      datestr = re.sub(r'([-+]\d{2}):(\d{2})(?:(\d{2}))?$', r'\1\2\3', datestr)
    return (datetime.strptime(datestr,'%Y-%m-%d') - epoch).total_seconds() * 1000.0

class AlphaVantageService():
  def __init__(self, symbol, alphavantage_key):
    self.isDownloadData = True
    self.alphavantage_key = alphavantage_key
    self.symbol = symbol
    self.date_str_format = '%Y-%m-%d'
  
  def prepareStockData(self):

    ts = TimeSeries(key=self.alphavantage_key, output_format='pandas')
    csv_path = data_folder+f'{self.symbol}_daily.csv'
    ohclv_data = self._downloadData(csv_path, ts.get_daily, True, symbol=self.symbol, outputsize='full')

    ti = TechIndicators(key=self.alphavantage_key, output_format='pandas')

    # get sma 9, 20, 50, 200
    csv_path = data_folder+f'{self.symbol}_sma_9.csv'
    sma_9 = self._downloadData(csv_path, ti.get_sma, False, symbol=self.symbol, interval='daily', time_period=9)

    csv_path = data_folder+f'{self.symbol}_sma_50.csv'
    sma_50 = self._downloadData(csv_path, ti.get_sma, False, symbol=self.symbol, interval='daily', time_period=50)

    csv_path = data_folder+f'{self.symbol}_sma_200.csv'
    sma_200 = self._downloadData(csv_path, ti.get_sma, False, symbol=self.symbol, interval='daily', time_period=200)
    
    # get bbands, this gives me the 20 sma as well
    csv_path = data_folder+f'{self.symbol}_bb_20.csv'
    bb_20 = self._downloadData(csv_path, ti.get_bbands, True, symbol=self.symbol, interval='daily', time_period=20)
    
    # get rsi
    csv_path = data_folder+f'{self.symbol}_rsi.csv'
    rsi = self._downloadData(csv_path, ti.get_rsi, False, symbol=self.symbol, interval='daily', time_period=20)

    # get cci
    csv_path = data_folder+f'{self.symbol}_cci.csv'
    cci = self._downloadData(csv_path, ti.get_cci, False, symbol=self.symbol, interval='daily', time_period=20)
    
    # get obv
    csv_path = data_folder+f'{self.symbol}_obv.csv'
    obv = self._downloadData(csv_path, ti.get_obv, False, symbol=self.symbol, interval='daily')

  def _downloadData(self, csv_path, get_data_func, reverse, **kwargs):
    if self.isDownloadData:
      last_date = None
      if os.path.exists(csv_path):
        data = pd.read_csv(csv_path)
        last_date = data.iloc[-1][0]

      print('downloading to {}'.format(csv_path))
      data, metadata = get_data_func(**kwargs)
      time.sleep(20)

      # The format for daily ohclv is 3. Last Refreshed
      # While for indicators is 3: Last Refreshed
      try:
        last_refreshed = metadata['3. Last Refreshed']
      except:
        last_refreshed = metadata['3: Last Refreshed']
        pass

      if reverse:
        data = data.iloc[::-1]
      
      data.to_csv(csv_path, index = True)
        
      if last_refreshed == last_date:
        self.isDownloadData = False
    
    return pd.read_csv(csv_path).to_numpy()
