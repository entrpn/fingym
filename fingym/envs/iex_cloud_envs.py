import os
import sys

import pandas as pd
import numpy as np

from datetime import datetime, timedelta

from fingym.envs.spy_envs import SpyEnv

from iexfinance.stocks import get_historical_data

class IEXCloudDailyEnv(SpyEnv):
  def __init__(self, symbol, start, end, iex_token):
    self.symbol = symbol
    self.date_str_format = '%Y-%m-%d'
    self.start = start
    self.end = end
    self.iex_token = iex_token
    super().__init__()
  
  def _convert_date_str_to_datetime(self,datestr):
    if sys.version_info[1] <=5:
      datestr = re.sub(r'([-+]\d{2}):(\d{2})(?:(\d{2}))?$', r'\1\2\3', datestr)
    if type(datestr) == str:
      return datetime.strptime(datestr,self.date_str_format)
    else:
      return datestr
  
  def _load_data(self):
    """
    3. if it does exist check that is up to date with the provided `to` date and if it is then we are done
    4. If is not up to date with the 'to' date, download the missing data
    5. save it to csv and return the data.
    """
    data_file = self._get_data_file()
    tmp_file = data_file+'_temp'

    if os.path.isfile(data_file):
      df = pd.read_csv(data_file)
      df2 = df.iloc[[0, -1]]
      first_date = self._convert_date_str_to_datetime(df2.at[0,'date'])
      last_date = self._convert_date_str_to_datetime(df2.at[df2.index[-1],'date'])

      # Download the data missing from start to first_date and append to beginning of df
      if self.start < first_date:
        beginning_df = get_historical_data(self.symbol, start=self.start, end=(first_date - timedelta(days=1)), close_only=False, output_format='pandas', token=self.iex_token)
        beginning_df.to_csv(tmp_file, index = True)
        beginning_df = pd.read_csv(tmp_file)
        frames = [beginning_df,df]
        df = pd.concat(frames)
        df.to_csv(data_file, index = False)
        os.remove(tmp_file)

      if self.end > last_date:
        ending_df = get_historical_data(self.symbol, start=(last_date + timedelta(days=1)), end=self.end, close_only=False, output_format='pandas', token=self.iex_token)
        ending_df.to_csv(tmp_file, index = True)
        ending_df = pd.read_csv(tmp_file)
        frames = [df, ending_df]
        df = pd.concat(frames)
        df.to_csv(data_file, index = False)
        os.remove(tmp_file)
      
      df['date'] = pd.to_datetime(df['date'])

      mask = (df['date'] >= self.start) & (df['date'] <=self.end)
      df = df.loc[mask]

    else:
      df = get_historical_data(self.symbol, start=self.start, end=self.end, close_only=False, output_format='pandas', token=self.iex_token)
      df.to_csv(data_file, index = True)

    return df.columns.values, df.to_numpy()

  def _get_data_file(self):
    return os.path.join(os.path.dirname(__file__),'..',f'data/iex_{self.symbol}_daily.csv')