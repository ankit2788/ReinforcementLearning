import pandas as pd
import numpy as np
import os
import re

from RLLibrary.utils import constants
DATA_DIR = constants.DATA_DIR
#DATA_DIR = os.path.join(DATA_DIR, "OrderExecution")

# MONTH mapping
MONTH = { 
    "JAN": "01",
    "FEB": "02",
    "MAR": "03",
    "APR": "04",
    "MAY": "05",
    "JUN": "06",
    "JUL": "07",
    "AUG": "08",
    "SEP": "09",
    "OCT": "10",
    "NOV": "11",
    "DEC": "12"

}

from RLLibrary.utils.loggingConfig import logger
Logger = logger.getLogger("DataManager")


class Dummy():
    def __init__(self):
        pass

class DataMatrices():

    def __init__(self, ticker, \
                filepath = os.path.join(DATA_DIR, "OrderExecution")):


        self.ticker = ticker
        self.dataPath = filepath

        
                
    def loadData(self, years = ["2018", "2019", "2020"], startTime = "09:16", endTime = "15:30"):

        columns = ["Ticker", "Date", "Time", "Open", "High", "Low", "Close", "Volume", "Temp"]

        # loads the data into a dataframe object
        self.data = {}
        self.dates = {}


        # get all years
        # years = []
        # for year in os.listdir(self.dataPath):
        #     if os.path.isdir(os.path.join(self.dataPath, year)):
        #         years.append(year)


        for year in years:
            self.data[year] = {}
            self.dates[year] = []

            folder = os.path.join(os.path.join(self.dataPath, year), self.ticker)
            print(f"Loading Data for {year}")

            for item in os.listdir(folder):

                try:

                    _date = item.split("_")[1].split(".")[0]

                    # convert date to YYYYMMDD format
                    details = re.split('([0-9]+)([A-Z]{3})([0-9]{4})$', _date.upper())
                    _day = details[1] if len(details[1]) == 2 else f'0{details[1]}'
                    _month = MONTH[details[2]]
                    _year = details[3]

                    _date = f'{_year}{_month}{_day}'

                    _data = pd.read_csv(os.path.join(folder, item), header = None, names = columns)
                    _data = _data.query('Time >= @startTime and Time <= @endTime')

                    if len(_data) > 0:
                        # get only OHLC
                        _times  = np.array(_data[["Time"]])
                        _prices = np.array(_data[["Open", "High", "Low", "Close"]])
                        _volume = np.array(_data[["Volume"]])

                        self.data[year][_date] = (_times, _prices, _volume)
                        self.dates[year].append(int(_date))



                except:
                    Logger.error(f'{_date} Data not loaded')

            self.dates[year] = np.sort(self.dates[year])


    def getVolumeStats(self):

        volume = []
        for year in self.data.keys():

            for date in self.data[year].keys():
                _thisDateVol = self.data[year][date][2]         # in a tuple of 3 elements, index 2 is volume
                _thisDateVol = list(_thisDateVol.reshape(_thisDateVol.shape[0], ))

                volume = volume + _thisDateVol

        # get VolumeStats
        self.VolumeStats = Dummy()

        self.VolumeStats.mean   = np.mean(volume)
        self.VolumeStats.std    = np.std(volume)
        self.VolumeStats.Data   = np.array(volume)





