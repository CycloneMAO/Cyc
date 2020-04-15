# Established by Jianmin MAO 
# Phone/WeChat: 18194038783
# QQ: 877245759

"""
This file contains StockData class which is a simple API designed for daily-based stock data
"""
import numpy as np
import pandas as pd
import os


class StockData:
    def __init__(self, path):
        self.path = path
        self.file_list = os.listdir(self.path)
        self.symbol_list = [i[:6] for i in self.file_list]
        self.raw_data = {}
        self.adjusted = {}

    def read(self, symbols: list):
        for symbol in symbols:
            if symbol in self.symbol_list:
                symbol_file = pd.Series(self.file_list)[[i == symbol for i in self.symbol_list]].values[0]
                self.raw_data[symbol] = pd.read_csv(self.path + "\\" + symbol_file).sort_values(by=['TRADE_DT'])
                # this set of indices is used for the date arguments in self.get_data_by_symbol() and self.get_data_by_date()
                self.raw_data[symbol].index = self.raw_data[symbol]['TRADE_DT']
                self.adjusted[symbol] = False
                print("load data of " + symbol + " done!")
            else:
                raise FileNotFoundError("File for "+symbol+" not found! Please check it!")

    def get_data_by_symbol(self, symbol, start_date, end_date):
        if symbol not in list(self.raw_data.keys()):
            raise NotImplementedError("Please load data for " + symbol + " first using self.read([symbols])!")
        else:
            data = self.raw_data[symbol].copy(deep=True)
            remain_index = data.index[data.index <= end_date]
            remain_index = remain_index.intersection(data.index[data.index >= start_date])
            data = data.loc[remain_index, ['TRADE_DT', 'S_DQ_OPEN', 'S_DQ_HIGH', 'S_DQ_LOW', 'S_DQ_CLOSE']]
            data.set_index('TRADE_DT', inplace=True)

            return data

    def get_data_by_date(self, adate: int, symbols):
        not_exist = []
        for symbol in symbols:
            if symbol not in list(self.raw_data.keys()):
                not_exist.append(symbol)
        if len(not_exist) != 0:
            print(not_exist)
            raise NotImplementedError("Please load data for the above symbols first using self.read([symbols])!")
        else:
            data = pd.DataFrame()
            for symbol in symbols:
                data_symbol = self.raw_data[symbol].copy(deep=True)
                data = pd.concat([data, data_symbol.loc[data_symbol.index == adate, :]], ignore_index=True)
            data = data.loc[:, ['S_INFO_WINDCODE', 'S_DQ_OPEN', 'S_DQ_HIGH', 'S_DQ_LOW', 'S_DQ_CLOSE']]
            data['S_INFO_WINDCODE'] = symbols
            data.set_index('S_INFO_WINDCODE', inplace=True)

            return data

    """Why don't I use pivot table functionality? I find that if the symbol list is large, 
        I have to combine their DataFrames to form a huge DataFrame so that the huge DataFrame 
        will result in slow speed and some memory problems.
        Hence I choose to generate the series for each symbol iteratively, then use them to reform the final DataFrame
    """
    def get_data_by_field(self, field, symbols):
        not_exist = []
        for symbol in symbols:
            if symbol not in list(self.raw_data.keys()):
                not_exist.append(symbol)
        if len(not_exist) != 0:
            print(not_exist)
            raise NotImplementedError("Please load data for the above symbols first using self.read([symbols])!")
        else:
            data_field = {}
            for symbol in symbols:
                data_symbol = self.raw_data[symbol].copy(deep=True)
                # automatically set 'TRADE_DT' as the index of the output DataFrame
                data_symbol.set_index('TRADE_DT', inplace=True)
                data_field[symbol] = data_symbol.loc[:, field].sort_index()
            data = pd.DataFrame(data_field)

            return data

    def format_date(self, symbol):
        symbol_data = self.raw_data[symbol]
        symbol_data['TRADE_DT'] = [pd.Timestamp(str(symbol_data['TRADE_DT'].iloc[i])) for i in range(symbol_data.shape[0])]

    def plot(self, symbol, field):
        self.format_date(symbol)
        if symbol == 'S_DQ_VOLUME' or symbol == 'S_DQ_AMOUNT':
            self.get_data_by_field(field, [symbol]).plot(kind='hist', title=field)
        else:
            self.get_data_by_field(field, [symbol]).plot(title=field)

    # Since the forward adjust(后复权) has already existed in the data, we only design a function to do backward adjust(前复权)
    def adjust_data(self, symbol):
        if not self.adjusted[symbol]:
            symbol_data = self.raw_data[symbol]  # no copy, imply possible modifications of the stored data
            adjust_fields = ['S_DQ_OPEN', 'S_DQ_HIGH', 'S_DQ_LOW', 'S_DQ_CLOSE', 'S_DQ_PRECLOSE', 'S_DQ_CHANGE', 'S_DQ_AVGPRICE']
            last_adjust = symbol_data['S_DQ_ADJFACTOR'].iloc[-1]
            for field in adjust_fields:
                symbol_data[field] = symbol_data[field]*symbol_data['S_DQ_ADJFACTOR']/last_adjust
            symbol_data['S_DQ_VOLUME_FWDADJ'] = symbol_data['S_DQ_VOLUME']/(symbol_data['S_DQ_ADJFACTOR']/last_adjust)
            self.adjusted[symbol] = True
        else:
            raise NotImplementedError("The data has been already backward-adjusted. Please don't do it again!")

    def resample(self, symbol, freq):  # use the data after backward adjustment(前复权)
        data = self.raw_data[symbol].copy(deep=True)
        pseudo_dates = pd.date_range(data['TRADE_DT'].iloc[0], periods=data.shape[0], freq='D')
        data.index = pseudo_dates
        resample_fields = {}
        resample_fields['S_DQ_OPEN'] = data['S_DQ_OPEN'].resample(str(freq) + 'D').first().to_numpy()
        resample_fields['S_DQ_CLOSE'] = data['S_DQ_CLOSE'].resample(str(freq) + 'D').last().to_numpy()
        resample_fields['S_DQ_HIGH'] = data['S_DQ_HIGH'].resample(str(freq) + 'D').max().to_numpy()
        resample_fields['S_DQ_LOW'] = data['S_DQ_LOW'].resample(str(freq) + 'D').min().to_numpy()
        resample_fields['S_DQ_AMOUNT'] = data['S_DQ_AMOUNT'].resample(str(freq) + 'D').sum().to_numpy()
        # I think volume should also be adjusted so that they will be consistent within each cycle.
        resample_fields['S_DQ_VOLUME'] = data['S_DQ_VOLUME_FWDADJ'].resample(str(freq) + 'D').sum().to_numpy()
        # To compute resampled vwap, I use backward-adjusted volume.
        resample_fields['S_DQ_AVGPRICE'] = resample_fields['S_DQ_AMOUNT'] / resample_fields['S_DQ_VOLUME'] * 10
        # use the last day of each cycle as the index to avoid using future data
        resample_index = data['TRADE_DT'].resample(str(freq) + 'D').last()
        resample_data = pd.DataFrame(resample_fields, index=resample_index)

        return resample_data

    def moving_average(self, symbol, field, window, add=True):
        field_data = self.get_data_by_field(field, [symbol])[symbol]
        result = field_data.rolling(window).mean()
        if add:
            self.raw_data[symbol][field+'_MA_'+str(window)] = result.to_numpy()

        return result

    # use close price to compute ema indicator
    def ema(self, symbol, alpha=0.5, span=None, add=True):
        symbol_data = self.raw_data[symbol]['S_DQ_CLOSE'].copy(deep=True)
        if span is None:
            close_ema = symbol_data.ewm(alpha=alpha).mean()
        else:
            close_ema = symbol_data.ewm(span=span).mean()
        close_ema.index = self.raw_data[symbol]['TRADE_DT']
        if add:
            if span is None:
                self.raw_data[symbol]['EMA_ALPHA_'+str(alpha)] = close_ema.to_numpy()
            else:
                self.raw_data[symbol]['EMA_SPAN_'+str(span)] = close_ema.to_numpy()

        return close_ema

    def atr(self, symbol, n, add=True):
        data_need = self.raw_data[symbol].loc[:, ['S_DQ_HIGH', 'S_DQ_LOW', 'S_DQ_PRECLOSE']].copy(deep=True)
        compare_series = {1: data_need['S_DQ_HIGH'] - data_need['S_DQ_LOW'],
                          2: abs(data_need['S_DQ_HIGH'] - data_need['S_DQ_PRECLOSE']),
                          3: abs(data_need['S_DQ_PRECLOSE'] - data_need['S_DQ_LOW'])}
        compare_data = pd.DataFrame(compare_series)
        result = compare_data.max(axis=1).rolling(n).mean()
        if add:
            self.raw_data[symbol]['ATR_'+str(n)] = result.to_numpy()

        return result

    def rsi(self, symbol, n, add=True):
        data_need = self.raw_data[symbol].loc[:, ['S_DQ_CLOSE', 'S_DQ_PRECLOSE']].copy(deep=True)
        every_period_gain = np.maximum(data_need['S_DQ_CLOSE']-data_need['S_DQ_PRECLOSE'], 0)
        every_period_deviate = abs(data_need['S_DQ_CLOSE']-data_need['S_DQ_PRECLOSE'])
        average_gain = every_period_gain.rolling(n).mean()
        average_loss = every_period_deviate.rolling(n).mean()
        result = average_gain/average_loss*100
        if add:
            self.raw_data[symbol]['RSI_'+str(n)] = result.to_numpy()
        return result

    # analogous to the acceleration of the price
    def macd(self, symbol, long, short, mid, add=True):
        dif = self.ema(symbol, span=short, add=False) - self.ema(symbol, span=long, add=False)
        dea = dif.ewm(span=mid).mean()
        result = (dif - dea) * 2
        if add:
            self.raw_data[symbol]['MACD_'+str(long)+'_'+str(short)+'_'+str(mid)] = result.to_numpy()
        return result