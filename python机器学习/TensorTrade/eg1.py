#!/usr/bin/env python
# coding: utf-8

# !pip install git+git://github.com/tensortrade-org/tensortrade.git

# In[217]:


import ta
import ssl
import pandas as pd


ssl._create_default_https_context = ssl._create_unverified_context # Only used if pandas gives a SSLError


def fetch(exchange_name, symbol, timeframe):
    url = "https://www.cryptodatadownload.com/cdd/"
    filename = "{}_{}USD_{}".format(exchange_name, symbol, timeframe)
    df = pd.read_csv('https://www.cryptodatadownload.com/cdd/Coinbase_BTCUSD_1h.csv', skiprows=1)
    df = df[::-1]
    df = df.drop(["Symbol", "Volume BTC"], axis=1)
    df = df.rename({"Volume USD": "volume"}, axis=1)
    df.columns = [name.lower() for name in df.columns]
    df = df.set_index("date")
    df.columns = [symbol + ":" + name.lower() for name in df.columns]
    return df


# In[ ]:
ticker='1'
time_interval=['20190102','20200123']
file_path=r'E:\Siwei\tensortrade_data\2019price_volume'
# In[ ]:
import os
def getSingleStockData(ticker:str,time_interval:[start,end],file_path:str):
  file_list=os.listdir(file_path)
  data_days=[]
  start=time_interval[0]
  end=time_interval[1]
  for i in range(len(file_list)):
    #i=0
    if int(file_list[i][:8])<=int(end) and int(file_list[i][:8])>=int(start):
      data_day=pd.read_csv(file_path+'\\'+file_list[i],converters={'#TICKER': lambda x: str(x),'DATE_':lambda x: str(x)},encoding='ISO-8859-1')
      stock_data=data_day.loc[data_day['#TICKER']==ticker,['DATE_','OPEN','HIGH','LOW','CLOSE','VOLUME','VWAP']]
      if stock_data.shape[0]==1:
        data_days.append(stock_data.iloc[0,:])
  result=pd.DataFrame(data_days)
  result.index=result['DATE_']
  result.drop('DATE_',axis=1,inplace=True)
  return result
# In[ ]:
pinan=getSingleStockData(ticker, time_interval, file_path)
# In[ ]:
# In[ ]:
# In[ ]:
# In[ ]:
# In[ ]:
# coinbase_data = pd.concat([
#     fetch("Coinbase", "BTC", "1h"),
#     fetch("Coinbase", "ETH", "1h")
# ], axis=1)

# bitstamp_data = pd.concat([
#     fetch("Bitstamp", "BTC", "1h"),
#     fetch("Bitstamp", "ETH", "1h"),
#     fetch("Bitstamp", "LTC", "1h")
# ], axis=1)




# In[213]:
# define the exchanges


from tensortrade.exchanges import Exchange
from tensortrade.exchanges.services.execution.simulated import execute_order
from tensortrade.data import Stream
# In[213]:
one_stock_exchange = Exchange("OSE", service=execute_order)(
    Stream("CNY-平安银行", list(pinan['CLOSE'])),
    #Stream("CNY-ETH", list(coinbase_data['ETH:close']))
)

# bitstamp = Exchange("bitstamp", service=execute_order)(
#     Stream("USD-BTC", list(bitstamp_data['BTC:close'])),
#     Stream("USD-ETH", list(bitstamp_data['ETH:close'])),
#     Stream("USD-LTC", list(bitstamp_data['LTC:close']))
# )


# In[78]:


# In[214]:


import ta

from tensortrade.data import DataFeed,Module

# Add all features for coinbase bitcoin
#coinbase_btc = coinbase_data.loc[:, [name.startswith("BTC") for name in coinbase_data.columns]]

ta.add_all_ta_features(
    pinan,
    colprefix="平安银行",
    **{'open': 'OPEN', 
       'high': 'HIGH',
       'low': 'LOW',
       'close': 'CLOSE',
       'volume':'VOLUME'}
)


with Module("平安银行") as pinan_ns:
    nodes = []
    nodes += [Stream(name, list(pinan[name])) for name in pinan.columns]
    
# Namespaces are used to give context to features so they don't 
# conflict with features of the same name that are being used for
# different data (e.g. different exchange)

# # Add all features for coinbase ethereum
# bitstamp_eth = bitstamp_data.loc[:, [name.startswith("ETH") for name in bitstamp_data.columns]]  
# ta.add_all_ta_features(
#     bitstamp_eth,
#     colprefix="ETH:",
#     **{k: "ETH:" + k for k in ['open', 'high', 'low', 'close', 'volume']}
# )

# nodes = []
# for name in coinbase_btc.columns:
#     nodes += [Stream(name, list(coinbase_btc[name]))]
    
# bitstamp_ns = Namespace("coinbase")(*nodes)

feed = DataFeed([pinan_ns])



# In[210]:


feed.next()


# In[215]:


from tensortrade.instruments import Instrument,USD, BTC, ETH, LTC,CNY
from tensortrade.wallets import Wallet, Portfolio
N000001 = Instrument('平安银行', 2, '000001')

portfolio = Portfolio(CNY, [
    Wallet(one_stock_exchange, 100000 * CNY),
    Wallet(one_stock_exchange, 100 * N000001)
])


# In[87]:


feed.next()


# In[89]:


feed.reset()


# In[206]:



from tensortrade.instruments import USD, BTC, ETH, LTC
from tensortrade.wallets import Wallet, Portfolio

portfolio = Portfolio(USD, [
    Wallet(coinbase, 10000 * USD),
    Wallet(coinbase, 10 * BTC),
    Wallet(coinbase, 5 * ETH),
    Wallet(bitstamp, 1000 * USD),
    Wallet(bitstamp, 5 * BTC),
    Wallet(bitstamp, 3 * LTC),
])


# In[216]:


from tensortrade.environments import TradingEnvironment

env = TradingEnvironment(portfolio=portfolio,
    action_scheme='managed-risk',
    reward_scheme='simple',
    feed=feed)


# In[137]:


# from tensortrade.data.internal.helpers import create_internal_feed


# # In[197]:


# new_feed=create_internal_feed(portfolio)  ###产生新环境时，除了传递给环境的feed外，还会自动生成一个内置feed，这个内置feed有问题，无法使用.next()


# In[ ]:





# In[198]:


# new_feed.next()


# In[183]:


# new_feed._next(new_feed.inputs[0])


# # In[199]:


# for i in range(len(new_feed.inputs)):
#     print(i)
#     new_feed._next(new_feed.inputs[i])  ###internal feed 的第三个node有问题


# # In[201]:


# i


# In[204]:


# for output_node in new_feed.inputs[1].outbound:
#     new_feed.inputs[1]._next(output_node)


# # In[187]:


# new_feed._data


# # In[148]:


# for i in range(len(new_feed.inputs)):
#     print(i,new_feed.inputs[i].forward(new_feed.inputs[i]._inbound_data))
    
    
    


# # In[166]:


# feed.inputs


# # In[167]:


# for node in feed.inputs:
#     feed._next(node)


# # In[175]:


# feed._next(feed.inputs[0].outbound[0])


# # In[176]:


# feed._data


# # In[161]:


# ll={}
# ll.update({'nsg':25})


# # In[162]:


# ll


# # In[178]:


# feed.inputs[:2]+feed.inputs[2:4]

# In[178]:
# In[178]:
# In[178]:
# In[178]:
# In[178]:
# In[178]:
# In[178]:
# In[178]:
# In[178]:
# In[178]:
# In[178]: