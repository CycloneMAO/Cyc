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
    def __init__(self, path: str) -> None:
        """
        arguments:
            path:  the underlying data path we're gonna work on
        attributes:
            file_list:  The list of file names in the directory of the input path
            symbol_list:    Extract the first 6 elements of each name in file_list to represent the symbols
            raw_data:   An dictionary including the data we load using self.read(symbol). raw_data[symbol]=data of this symbol
            adjusted:   In the computation of financial indicators, we'll use backward-adjusted data.
                        Once the raw_data has been adjusted, adjusted[symbol]=True and no more adjustment can be done to this symbol.
                        It prevents the data from multiple adjustment
        """
        self.path = path
        self.file_list = os.listdir(self.path)
        self.symbol_list = [i[:6] for i in self.file_list]
        self.raw_data = {}
        self.adjusted = {}

    def read(self, symbols: list) -> None:
        """
        :param symbols: a list of symbols of which you want to load data.
        :return: None. The data will be stored in self.raw_data[symbol] for each symbol in the list, symbols
        """
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

    def get_data_by_symbol(self, symbol: str, start_date: int, end_date: int) -> pd.DataFrame:
        """
        :param symbol: a string indicating what symbol of the data to load. eg: '000001'
        :param start_date: an 8-digit integer representing the starting date of the data to load. eg: 19990302
        :param end_date:  ending date of the data to load. eg: 20010102
        :return: a pandas DataFrame object containing the data
        """
        if symbol not in list(self.raw_data.keys()):
            raise NotImplementedError("Please load data for " + symbol + " first using self.read([symbols])!")
        else:
            data = self.raw_data[symbol].copy(deep=True)
            remain_index = data.index[data.index <= end_date]
            remain_index = remain_index.intersection(data.index[data.index >= start_date])
            data = data.loc[remain_index, ['TRADE_DT', 'S_DQ_OPEN', 'S_DQ_HIGH', 'S_DQ_LOW', 'S_DQ_CLOSE']]
            data.set_index('TRADE_DT', inplace=True)

            return data

    def get_data_by_date(self, adate: int, symbols: list) -> pd.DataFrame:
        """
        :param adate: an 8-digit integer representing the date of the data to load. eg: 19990323.
                        If adate doesn't exist for any symbol, the data of that symbol won't appear in the result
        :param symbols: a list of symbols of the data to load. eg: ['000001', '000002']
        :return: a pandas DataFrame object containing the data to load
        """
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
    def get_data_by_field(self, field: str, symbols: list) -> pd.DataFrame:
        """
        :param field: a string representing the field of the data to load. It matches the columns in self.raw_data[symbol]. eg: "S_DQ_OPEN'
        :param symbols: a list of symbols
        :return: a pandas DataFrame
        """
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

    def format_date(self, symbol: str) -> None:
        """
        :param symbol: a string representing the symbol of date to be formatted. eg: '000001'
        :return: None. The column "TRADE_DT' in self.raw_data[symbol] will be transformed to pd.Timestamp object
        """
        symbol_data = self.raw_data[symbol]
        symbol_data['TRADE_DT'] = [pd.Timestamp(str(symbol_data['TRADE_DT'].iloc[i])) for i in range(symbol_data.shape[0])]

    def plot(self, symbol: str, field: str) -> None:
        """
        :param symbol: a string representing the symbol of the data to visualize. eg: '000001'
        :param field: a string specifying the field to plot.
        :return: None. A plot will be rendered. Recommend to use PyCharm IDE
        """
        self.format_date(symbol)
        if symbol == 'S_DQ_VOLUME' or symbol == 'S_DQ_AMOUNT':
            self.get_data_by_field(field, [symbol]).plot(kind='hist', title=field)
        else:
            self.get_data_by_field(field, [symbol]).plot(title=field)

    # Since the forward adjust(后复权) has already existed in the data, we only design a function to do backward adjust(前复权)
    def adjust_data(self, symbol: str) -> None:
        """
        :param symbol: a str representing the symbol of stock data to adjust.
                        If the data has been adjusted, will raise NotImplementedError
        """
        if not self.adjusted[symbol]:
            symbol_data = self.raw_data[symbol]  # no copy, imply possible modifications of the stored data
            adjust_fields = ['S_DQ_OPEN', 'S_DQ_HIGH', 'S_DQ_LOW', 'S_DQ_CLOSE', 'S_DQ_PRECLOSE', 'S_DQ_CHANGE', 'S_DQ_AVGPRICE']
            last_adjust = symbol_data['S_DQ_ADJFACTOR'].iloc[-1]
            for field in adjust_fields:
                symbol_data[field] = symbol_data[field]*symbol_data['S_DQ_ADJFACTOR']/last_adjust
            # I think volume should also be adjusted so that they will be consistent within each cycle.
            symbol_data['S_DQ_VOLUME_FWDADJ'] = symbol_data['S_DQ_VOLUME']/(symbol_data['S_DQ_ADJFACTOR']/last_adjust)
            self.adjusted[symbol] = True
        else:
            raise NotImplementedError("The data has been already backward-adjusted. Please don't do it again!")

    def resample(self, symbol: str, freq: int) -> pd.DataFrame:
        """
        :param symbol: a symbol of data which has been adjusted using self.adjust_data(symbol).
                        If not adjusted yet, will adjust it first before re_sampling
        :param freq: the size of the re_sampling window
        :return: a pandas DataFrame containing re_sampled data
        """
        if not self.adjusted[symbol]:
            self.adjust_data(symbol)
        data = self.raw_data[symbol].copy(deep=True)
        # Here I create a pseudo dates object to leverage pandas.resample() functionality to help calculation
        pseudo_dates = pd.date_range(data['TRADE_DT'].iloc[0], periods=data.shape[0], freq='D')
        data.index = pseudo_dates
        resample_fields = {'S_DQ_OPEN': data['S_DQ_OPEN'].resample(str(freq) + 'D').first().to_numpy(),
                           'S_DQ_CLOSE': data['S_DQ_CLOSE'].resample(str(freq) + 'D').last().to_numpy(),
                           'S_DQ_HIGH': data['S_DQ_HIGH'].resample(str(freq) + 'D').max().to_numpy(),
                           'S_DQ_LOW': data['S_DQ_LOW'].resample(str(freq) + 'D').min().to_numpy(),
                           'S_DQ_AMOUNT': data['S_DQ_AMOUNT'].resample(str(freq) + 'D').sum().to_numpy(),
                           'S_DQ_VOLUME': data['S_DQ_VOLUME_FWDADJ'].resample(str(freq) + 'D').sum().to_numpy()}
        # To compute resampled vwap, I use backward-adjusted volume, i.e., 'S_DQ_VOLUME_FWDADJ' in the adjusted data.
        resample_fields['S_DQ_AVGPRICE'] = resample_fields['S_DQ_AMOUNT'] / resample_fields['S_DQ_VOLUME'] * 10
        # use the last day of each cycle as the index to avoid using future data
        resample_index = data['TRADE_DT'].resample(str(freq) + 'D').last()
        resample_data = pd.DataFrame(resample_fields, index=resample_index)

        return resample_data

    def moving_average(self, symbol: str, field: str, window: int, add: bool = True) -> pd.Series:
        """
        :param symbol: the symbol to implement moving average. As a mandate, using backward-adjusted price.
                        If the data of the symbol has not been adjusted, will adjust first using self.adjust_data(symbol)
        :param field: the field specified to implement moving average
        :param window: a positive integer representing the window_size of the resulting moving average series
        :param add: a boolean indicating whether to include this moving average to the stored self.raw_data[symbol].
                        Default is True so that one can plot this new field using self.plot() immediately .
        :return: a pandas Series object.
        """
        if not self.adjusted[symbol]:
            self.adjust_data(symbol)
        field_data = self.get_data_by_field(field, [symbol])[symbol]
        result = field_data.rolling(window).mean()
        if add:
            self.raw_data[symbol][field+'_MA_'+str(window)] = result.to_numpy()

        return result

    def ema(self, symbol: str, alpha: float = 0.5, span: int = None, add: bool = True) -> pd.Series:
        """
        :param symbol: The symbol to compute ema using its backward-adjusted close price as a mandate.
                        If the data of the symbol has not been adjusted, will adjust first using self.adjust_data(symbol)
        :param alpha: Specify smoothing factor, i.e., the weight assigned to the latest price of each window.
        :param span: Specify decay in terms of span
        :param add: a boolean indicating whether to include the ema of this symbol to the stored self.raw_data[symbol].
                        Default is True so that one can plot this new field using self.plot() immediately .
        :return: a pandas Series object
        """
        if not self.adjusted[symbol]:
            self.adjust_data(symbol)
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

    def atr(self, symbol: str, n: int, add: bool = True) -> pd.Series:
        """
        :param symbol: the symbol to compute atr using its backward-adjusted price as a mandate
                        If the data of the symbol has not been adjusted, will adjust first using self.adjust_data(symbol)
        :param n: size of the window of this indicator
        :param add: a boolean indicating whether to include the atr of this symbol to the stored self.raw_data[symbol].
                        Default is True so that one can plot this new field using self.plot() immediately .
        :return: a pandas Series object
        """
        if not self.adjusted[symbol]:
            self.adjust_data(symbol)
        data_need = self.raw_data[symbol].loc[:, ['S_DQ_HIGH', 'S_DQ_LOW', 'S_DQ_PRECLOSE']].copy(deep=True)
        compare_series = {1: data_need['S_DQ_HIGH'] - data_need['S_DQ_LOW'],
                          2: abs(data_need['S_DQ_HIGH'] - data_need['S_DQ_PRECLOSE']),
                          3: abs(data_need['S_DQ_PRECLOSE'] - data_need['S_DQ_LOW'])}
        compare_data = pd.DataFrame(compare_series)
        result = compare_data.max(axis=1).rolling(n).mean()
        if add:
            self.raw_data[symbol]['ATR_'+str(n)] = result.to_numpy()

        return result

    def rsi(self, symbol: str, n: int, add: bool = True) -> pd.Series:
        """
        :param symbol: the symbol to compute rsi using its backward-adjusted price as a mandate
                        If the data of the symbol has not been adjusted, will adjust first using self.adjust_data(symbol)
        :param n: size of the window of this indicator
        :param add: a boolean indicating whether to include the rsi of this symbol to the stored self.raw_data[symbol].
                        Default is True so that one can plot this new field using self.plot() immediately .
        :return: a pandas Series object
        """
        if not self.adjusted[symbol]:
            self.adjust_data(symbol)
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
    def macd(self, symbol: str, long: int, short: int, mid: int, add: bool = True) -> pd.Series:
        """
        :param symbol: the symbol to compute macd using its backward-adjusted price as a mandate
                        If the data of the symbol has not been adjusted, will adjust first using self.adjust_data(symbol)
        :param long: the size of the long-period window
        :param short:  the size of the short-period window
        :param mid: the size of the ema window to smoothing the intermediate dif series
        :param add: a boolean indicating whether to include the macd of this symbol to the stored self.raw_data[symbol].
                        Default is True so that one can plot this new field using self.plot() immediately .
        :return: a pandas Series object
        """
        if not self.adjusted[symbol]:
            self.adjust_data(symbol)
        dif = self.ema(symbol, span=short, add=False) - self.ema(symbol, span=long, add=False)
        dea = dif.ewm(span=mid).mean()
        result = (dif - dea) * 2
        if add:
            self.raw_data[symbol]['MACD_'+str(long)+'_'+str(short)+'_'+str(mid)] = result.to_numpy()
        return result