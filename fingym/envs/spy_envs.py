# Copyright 2020 The fingym Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pandas as pd
import numpy as np
import sys
from datetime import datetime
import re

from fingym.envs.env import Env
from fingym.spaces.space import BuyHoldSellSpace

class SpyEnv(Env):
    def __init__(self):

        self.headers, self.data = self._load_data()
        self.n_step = len(self.data)
        self.cur_step = 0
        self.initial_investment = 25000
        self.cash_in_hand = self.initial_investment
        self._close_idx = 4
        self._open_idx = 1
        self.stock_owned = 0

        # stock_owned, cash_in_hand, date, hclo, volume
        self.state_dim = 2 + len(self.headers)

        # action[0]
        # 0 - hold/do nothing
        # 1 - buy
        # 2 - hold
        # action[1]
        # Number of actions to buy/sell
        self.action_size = 2
        self.action_space = BuyHoldSellSpace()
    
    def reset(self):
        self.cur_step = 0
        self.cash_in_hand = self.initial_investment
        self.stock_owned = 0
        return self._get_obs()

    def step(self, action):
        
        prev_val = self._get_val()
        if self.cur_step < self.n_step -1:
            self.cur_step += 1
        # else:
        #     print('WARNING - this environment is done')
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
        obs[2] = self._convert_date_str_to_datetime(self.data[self.cur_step][0]).timestamp()
        obs[3:] = self.data[self.cur_step][1:]
        return obs

    def _get_val(self):
        return self.stock_owned * self._get_stock_price_close() + self.cash_in_hand
    
    def _get_stock_price_close(self):
        return self.data[self.cur_step][self._close_idx]

    def _get_stock_price_open(self):
        return self.data[self.cur_step][self._open_idx]

    def _load_data(self):
        raise NotImplementedError

    def _convert_date_str_to_datetime(self,datestr):
        raise NotImplementedError

class DailySpyEnv(SpyEnv):
    def __init__(self):
        self.date_str_format = '%m/%d/%Y'
        super().__init__()
    
    def _convert_date_str_to_datetime(self,datestr):
        if sys.version_info[1] <=5:
            datestr = re.sub(r'([-+]\d{2}):(\d{2})(?:(\d{2}))?$', r'\1\2\3', datestr)
        return datetime.strptime(datestr,self.date_str_format)
    
    def _load_data(self):
        data_file = self._get_data_file()
        df = pd.read_csv(data_file)
        return df.columns.values, df.values
    
    def _get_data_file(self):
        return os.path.join(os.path.dirname(__file__),'..','data/filtered_spy_data_10_yrs.csv')

class SpyDailyRandomWalkEnv(DailySpyEnv):
    def __init__(self, no_days_to_random_walk):

        self.no_days_to_random_walk = no_days_to_random_walk
        self.original_close = None

        super().__init__()

        self._close_idx = 1
        self._open_idx = self._close_idx
    
    def step(self, action):
        obs, reward, done, info = super().step(action)
        info['original_close'] = self._get_original_close_stock_price()
        return obs, reward, done, info

    def reset(self):
        self.headers, self.data = self._load_data()
        return super().reset()

    def _get_geometric_brownian_motion(self, so, mu, sigma):
        drift = (mu - 0.5 * sigma**2)
        diffusion = sigma * np.random.normal()
        return so*np.exp(drift + diffusion)
    
    def _load_data(self):
        data_file = self._get_data_file()
        df = pd.read_csv(data_file)
        df = df.drop(['open','high','low','volume'], axis = 1)
        
        df['daily_pct_change'] = df['close'].pct_change()
        mu = df['daily_pct_change'].iloc[:-self.no_days_to_random_walk].mean()
        sigma = df['daily_pct_change'].iloc[:-self.no_days_to_random_walk].std()

        today = df['close'].values[-self.no_days_to_random_walk]
        df['sim'] = df['close']
        df_len = df.shape[0]
        for days in range(1,self.no_days_to_random_walk):
            next_day = self._get_geometric_brownian_motion(today, mu, sigma)
            df["sim"][df_len-self.no_days_to_random_walk+days] = next_day
            today = next_day

        self._set_original_close_values(df['close'].values)

        df = df.drop(['close','daily_pct_change'], axis = 1)

        return df.columns.values, df.values

    def _set_original_close_values(self, original_close):
        self.original_close = original_close
    
    def _get_original_close_stock_price(self):
        return self.original_close[self.cur_step]

    def _get_obs(self):
        """
        Return array with number stock owned, stock price and cash in hand.
        Ex: [stock_owned, cash_in_hand, date, hclo, volume]
        """
        obs = np.empty(self.state_dim)
        obs[0] = self.stock_owned
        obs[1] = self.cash_in_hand
        # date
        obs[2] = self._convert_date_str_to_datetime(self.data[self.cur_step][0]).timestamp()
        obs[3] = self.data[self.cur_step][-1]
        return obs

class IntradaySpyEnv(SpyEnv):
    def __init__(self):
        self.date_str_format = '%Y-%m-%d %H:%M:%S%z'
        super().__init__()
    
    def _convert_date_str_to_datetime(self,datestr):
        if sys.version_info[1] <=5:
            datestr = re.sub(r'([-+]\d{2}):(\d{2})(?:(\d{2}))?$', r'\1\2\3', datestr)
        return datetime.strptime(datestr,self.date_str_format)
    
    def _load_data(self):
        data_file = os.path.join(os.path.dirname(__file__),'..','data/filtered_spy_2017_2019_all.csv')
        df = pd.read_csv(data_file)
        return df.columns.values, df.values