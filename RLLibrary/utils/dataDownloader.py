import pandas as pd
import numpy as np
import yfinance as yf       # yahoo finance
import logging
import os, sys


cwd = os.getcwd()

pref = os.path.dirname(os.path.realpath(__file__))
path = f'{cwd}/../'

os.environ["RL_PATH"]   = path

if f'{path}' not in sys.path:    
    sys.path.append(f'{path}')


if f'{path}/RLLibrary' not in sys.path:    
    sys.path.append(f'{path}/RLLibrary')


    

# import custom libraries
from RLLibrary.utils.loggingConfig import logger
from RLLibrary.utils import constants
Logger = logger.getLogger("DataIO")


highbeta = [            # Invesco S&P 500 High Beta ETF
    "PVH", 
    "OXY", 
    "NCLH", 
    "MGM", 
    "LNC",
    "HAL",
    "DVN",
    "DFS",
    "CCL",      
    "APA"	
]

lowbeta = [             # Invesco S&P 500® Low Volatility ETF
    "VZ",
    "COST",
    "JNJ",
    "BMY",
    "EA",
    "CERN",
    "CLX",
    "EXPD",
    "DG",
    "GIS"
]


DOWNLOAD_PATH = os.path.join(constants.DATA_DIR, "PortfolioManagement")


#download data
def downloader(ticker, startDate, endDate):
    # startDate and endDate --> format: YYYY-MM-DD
    
    Logger.info(f"Downloading data for {ticker}")
    data = yf.download(ticker, start=startDate, end=endDate)
    return data


def main(tickers, startDate, endDate):
    # startDate and endDate --> format: YYYY-MM-DD

    for ticker in tickers:
        
        _thisdata = downloader(ticker, startDate, endDate)
        
        # save file
        _savePath = os.path.join(DOWNLOAD_PATH, f'{ticker}.csv')
        _thisdata.to_csv(_savePath)
        
        
        
if __name__ == "__main__":
    
    if len(sys.argv) == 3:
        startDate = sys.argv[1]
        endDate = sys.argv[2]
        
    else:
        startDate = "2004-01-01"
        endDate = "2021-04-30"

    
    alltickers = highbeta + lowbeta
    main(alltickers, startDate, endDate )
    
        
        
        
        
        
        
    
    
    



    
    

