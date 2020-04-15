# import pandas as pd

# import os

# data = pd.read_csv(os.path.join(os.path.dirname(__file__),'amzn_full_weekly.csv'))
# data = data[['Date','Open','High','Low','Close','Volume']]
# data.set_index('Date',inplace=True)
# data.to_csv(os.path.join(os.path.dirname(__file__),'filtered_amzn_full_weekly.csv'))
# print(data.head)