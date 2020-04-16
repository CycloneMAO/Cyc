# Established by Jianmin MAO 
# Phone/WeChat: 18194038783
# QQ: 877245759

"""
This file contains all the testing for the functionality of StockData class
"""

from StockData import StockData

# Please go through all the sub-questions one by one and check the results
# E1.1  StockData.__init__(path)
data_path = r'.\training\data'
myDataManager = StockData(data_path)

# E1.2  StockData.read(symbols)
myDataManager.read(['000001', '000002'])

# E1.3  StockData.get_data_by_symbol(symbol, start_date, end_date)
myDataManager.get_data_by_symbol('000001', 19990301, 19990602).head(10)

# E1.4  StockData.get_data_by_date(adate, symbols):
myDataManager.get_data_by_date(19990323, ['000001', '000002'])

# E1.5  StockData.get_data_by_filed(field, symbols):
myDataManager.get_data_by_field('S_DQ_OPEN', ['000001', '000002'])


# E2.1 StockData.format_date(symbol)
myDataManager.format_date('000001')
myDataManager.raw_data['000001']['TRADE_DT']

# E2.2 StockData.plot(symbol, field)
myDataManager.plot('000001', 'S_DQ_OPEN')
myDataManager.plot('000002', 'S_DQ_CLOSE')
myDataManager.plot('000001', 'S_DQ_VOLUME')
myDataManager.plot('000002', 'S_DQ_AMOUNT')

# E2.3 StockData.adjust_data(symbol)
myDataManager.adjust_data('000001')
# after adjusting, we could see the curve is much smoother
myDataManager.plot('000001', 'S_DQ_OPEN')
myDataManager.plot('000001', 'S_DQ_VOLUME_FWDADJ')

# E2.4 StockData.resample(symbol, freq)
myDataManager.resample('000001', 11)


# E3.1 StockData.moving_average(symbol，field，window)
myDataManager.moving_average('000001', 'S_DQ_CLOSE', 5)
myDataManager.moving_average('000001', 'S_DQ_CLOSE', 20)
myDataManager.moving_average('000001', 'S_DQ_CLOSE', 60)
myDataManager.plot('000001', 'S_DQ_CLOSE_MA_5')
myDataManager.plot('000001', 'S_DQ_CLOSE_MA_20')
myDataManager.plot('000001', 'S_DQ_CLOSE_MA_60')

# E3.2 StockData.ema(symbol, params)
myDataManager.ema(symbol='000001', span=9)
myDataManager.ema(symbol='000001', alpha=0.2)
myDataManager.plot('000001', 'EMA_SPAN_9')
myDataManager.plot('000001', 'EMA_ALPHA_0.2')

# E3.3 StockData.atr(symbol, params)
myDataManager.atr('000001', 5)
myDataManager.atr('000001', 20)
myDataManager.atr('000001', 60)
myDataManager.plot('000001', 'ATR_5')
myDataManager.plot('000001', 'ATR_20')
myDataManager.plot('000001', 'ATR_60')

# E3.4 StockData.rsi(symbol, params)
myDataManager.rsi('000001', 5)
myDataManager.rsi('000001', 20)
myDataManager.rsi('000001', 60)
myDataManager.plot('000001', 'RSI_5')
myDataManager.plot('000001', 'RSI_20')
myDataManager.plot('000001', 'RSI_60')

# E3.5 StockData.macd(symbol, params)
myDataManager.macd('000001', 26, 12, 9)
myDataManager.plot('000001', 'MACD_26_12_9')
