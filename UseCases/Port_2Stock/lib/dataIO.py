import pandas as pd
import numpy as np
import yfinance as yf       # yahoo finance
from logging.config import dictConfig
import logging
import os, sys


cwd = os.getcwd()
path = f'{cwd}/../lib'

os.environ["PORT_MGMT_PATH"] = path

if os.environ["PORT_MGMT_PATH"] not in sys.path:
    sys.path.append(os.environ["PORT_MGMT_PATH"])

    
    

# import custom libraries
import configs
import loggingConfig


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

dictConfig(loggingConfig.DEFAULT_LOGGING)
Logger = logging.getLogger("DataIO")


#download data
def downloader(ticker, startDate, endDate):
    # startDate and endDate --> format: YYYY-MM-DD
    
    Logger.info(f"Downloading data for {ticker}")
    data = yf.download(ticker, start=startDate, end=endDate)
    return data["Adj Close"]


def main(tickers, startDate, endDate):
    # startDate and endDate --> format: YYYY-MM-DD

    for ticker in tickers:
        
        _thisdata = downloader(ticker, startDate, endDate)
        
        # save file
        _savePath = os.path.join(configs.DATA_DIR, f'{ticker}.csv')
        _thisdata.to_csv(_savePath)
        
        
        
if __name__ == "__main__":
    
    if len(sys.argv) == 3:
        startDate = sys.argv[1]
        endDate = sys.argv[2]
        
    else:
        startDate = "2004-01-01"
        endDate = "2020-12-25"

    
    alltickers = highbeta + lowbeta
    main(alltickers, startDate, endDate )
    
        
        
        
        
        
        
    
    
    



    
    

