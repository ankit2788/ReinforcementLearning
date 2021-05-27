import os
import shutil
import sys

def getData(ticker, year, path, \
    outputpath = '/Users/ankitgupta/Documents/git/anks/MachineLearning/ReinforcementLearning/data/OrderExecution/'):


    # create path if doesnt exist
    outputpath = os.path.join(outputpath, str(year))

    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
        
    # add stock path
    outputpath = os.path.join(outputpath, ticker)

    if not os.path.exists(outputpath):
        os.makedirs(outputpath)


    count = 0
    for subdir, dirs, files in os.walk(path):
        for file in files:
            if file == f"{ticker}.txt":
            #print(file)
                try:
                    date = int(subdir.split("/")[-1][:2])
                    month = subdir.split("/")[-1][2:]
                    count += 1
                    
                    #print(subdir)
                    filename = f'{ticker}_{date}{month}{year}.txt'

                    shutil.copy(os.path.join(subdir, file), os.path.join(outputpath, filename))
                    
                except:
                    pass
                
    print(f'{count} files saved for {ticker} in year {year}' )



if __name__ == "__main__":

    ticker = sys.argv[1]
    year = sys.argv[2]
    path = sys.argv[3]

    getData(ticker, year, path)




