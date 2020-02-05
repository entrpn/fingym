import pandas as pd
import numpy as np
import itertools
from datetime import datetime, timedelta
import re
import sys
import os
from .envs.spy_envs import DailySpyEnv

default_data_file = os.path.join(os.path.dirname(__file__),'data/filtered_spy_2017_2019_all.csv')

def make(envName, data_file=default_data_file):
    if envName == 'SPY-Daily-v0':
        return DailySpyEnv()
    return TickSpyEnv(data_file=data_file)

class TickSpyEnv():
    def __init__(self,data_file,initial_investment=25000):
        self.headers, self.data, self.data_timestamp = self._load_data(data_file)
        self.date_str_format='%Y-%m-%d %H:%M:%S%z'
        self.n_step = len(self.data)
        self.cur_step = 0
        self.initial_investment = initial_investment
        self.cash_in_hand = initial_investment
        self._close_idx = 3
        self.stock_owned = 0

        # date in ms, stock_owned + stock_technical_values + cash_in_hand
        self.state_dim = 2 + len(self.headers) + 1
        # 0 - hold/do nothing
        # 1 - buy
        # 2 - sell
        self.action_size = 3
    
    def reset(self):
        self.cur_step = 0
        self.cash_in_hand = self.initial_investment
        self.stock_owned = 0
        return self._get_obs()  

    def step(self, action):

        prev_val = self._get_val()

        self.cur_step += 1

        cur_val = self._get_val()

        self._trade(action)

        reward = cur_val - prev_val
        obs = self._get_obs()
        done = self.cur_step == self.n_step - 1
        info = {'cur_val': cur_val}

        return obs, reward, done, info

    def _trade(self,action):

        stock_price = self._get_stock_price()
        
        # sell
        shares_to_sell = 0
        if action[0] == 2:
            shares_to_sell = action[1]
        
        # buy
        shares_to_buy = 0
        if action[0] == 1:
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
        Ex: [timestamp,100,[data],11000]
        """
        obs = np.empty(self.state_dim)
        curr_time = self.data_timestamp[self.cur_step]
        obs[0] = self._convert_date_str_to_datetime(curr_time).timestamp()
        obs[1] = self.stock_owned
        obs[2:len(self.headers)+2] = self.data[self.cur_step]
        obs[-1] = self.cash_in_hand
        return obs
    
    def _get_val(self):
        return self.stock_owned * self._get_stock_price() + self.cash_in_hand

    def _get_stock_price(self):
        return self.data[self.cur_step][self._close_idx]
    
    def _load_data(self,data_file):
        df = pd.read_csv(data_file)
        timestamp = df['timestamp']
        df = df.drop(columns=['timestamp'])
        return df.columns.values, df.values, timestamp.values
    
    def _convert_date_str_to_datetime(self,datestr):
        if sys.version_info[1] <=5:
            datestr = re.sub(r'([-+]\d{2}):(\d{2})(?:(\d{2}))?$', r'\1\2\3', datestr)
        return datetime.strptime(datestr,self.date_str_format)