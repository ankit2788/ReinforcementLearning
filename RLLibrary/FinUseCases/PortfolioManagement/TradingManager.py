import numpy as np


class Trader():

    def __init__(self, TransactionFee):

        self.TransactionFee = TransactionFee




    def performTrade(self, currentCash, price, units):
        # with price and units, compute transaction cost and revised cash



        # udpate cash
        units = np.array(units)
        price = np.array(price)

        _effectiveCash = np.multiply(units, price)

        transactionCost = np.sum(np.abs(_effectiveCash)) * self.TransactionFee
        transactionCost = np.round(transactionCost, 2)

        updatedCash = np.round(currentCash - np.sum(_effectiveCash) - transactionCost, 2)

        return updatedCash, transactionCost
