import os
import pandas as pd
import numpy as np

from RLLibrary.utils import constants
DATA_DIR = constants.DATA_DIR


class DataMatrices():

    def __init__(self, assets, startDate, endDate, filepath = os.path.join(DATA_DIR, "PortfolioManagement")):

        self.nbAssets = len(assets)
        self.assets = assets
        self.startDate = startDate
        self.endDate = endDate
        self.filepath = filepath


    def loadData(self, startDate = None, endDate = None, filepath = os.path.join(DATA_DIR, "PortfolioManagement")):

        dataPath = filepath if filepath is not None else self.filepath
        startDate = startDate if startDate is not None else self.startDate
        endDate = endDate if endDate is not None else self.endDate

        requiredCols = ["High", "Low", "Adj Close"]


        data_close      = pd.DataFrame()
        data_high       = pd.DataFrame()
        data_low        = pd.DataFrame()
        data_returns    = pd.DataFrame()

        for asset in self.assets:
            _path       = os.path.join(dataPath, f"{asset}.csv")
            _thisasset  = pd.read_csv(_path, index_col = "Date")

            if startDate is not None and endDate is not None:
                _thisasset = _thisasset[(_thisasset.index >= startDate) & (_thisasset.index <= endDate)]   
            
            _thisasset = _thisasset[requiredCols]                        
            _thisasset["Return"] = _thisasset["Adj Close"].pct_change() * 100   # convert to percent change         

            #columns = ["Adj Close", "Return"]
            data_close      = data_close.merge(_thisasset[["Adj Close"]], how = "outer", left_index = True, right_index = True)
            data_returns    = data_returns.merge(_thisasset[["Return"]], how = "outer", left_index = True, right_index = True)
            data_high       = data_high.merge(_thisasset[["High"]], how = "outer", left_index = True, right_index = True)
            data_low        = data_low.merge(_thisasset[["Low"]], how = "outer", left_index = True, right_index = True)

            data_close.rename(columns={'Adj Close': f'Price_{asset}'}, inplace=True)
            data_returns.rename(columns={'Return': f'Ret_{asset}'}, inplace=True)

            data_high.rename(columns={'High': f'High_{asset}'}, inplace=True)
            data_low.rename(columns={'Low': f'Low_{asset}'}, inplace=True)
            
            
        # fillNAs if any
        data_close = data_close.fillna(method = "ffill")            
        data_high = data_high.fillna(method = "ffill")            
        data_low = data_low.fillna(method = "ffill")            
        data_returns = data_returns.fillna(method = "ffill")            


        # convert the data into an array form
        shape = (len(data_close), self.nbAssets, 4)
        data = np.zeros(shape)

        data[:,:,0] = data_close.values             # close values      
        data[:,:,1] = data_high.values              # hgh values
        data[:,:,2] = data_low.values               # low vales
        data[:,:,3] = data_returns.values           # daily returns

        dates = np.array(data_close.index)

        self.dates = dates
        self.data = data



    def getDataForDate(self, date):

        # get data for a particular date
        _index = np.where(self.dates == date)

        if len(_index) == 1:
            return self.data[_index[0]]
        return None                

    def getDataForRange(self, startIndex, endIndex):
        return self.data[startIndex:endIndex+1], self.dates[startIndex:endIndex+1]


