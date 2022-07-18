### Getting the dicts for apps ###

import numpy as np
import pandas as pd
import openpyxl

class Dicts:
    def __init__(self):
        self.wb_obj = openpyxl.load_workbook("C:/Users/lgarcia/OneDrive - CUSTOMSOFT/DATABASE.xlsx")

    def stocks_dict(self):
        # Getting the index columns for sectors
        a = range(1,56)
        b = []
        for i in a:
            if i%5==0:
                b.append(i-1)

        # Getting the list of sectors
        sectors=[sector for sector in pd.DataFrame(self.wb_obj["SECTOR"].values)[b].iloc[1,:]]

        # Getting the index columns for symbols
        a = range(1,56)
        b = []
        for i in a:
            if i%5==0:
                b.append(i-3)

        # Getting the list of symbols
        symbols = []
        for i in b:
            symbols.append([a for a in list(pd.DataFrame(self.wb_obj["SECTOR"].values)[i].to_numpy()) if a not in [None,"Symbol"]])

        # Creating the dictionary
        return dict(zip(sectors, symbols))

    def etf_dict(self):
        # Getting the ETF dict
        symbols = [symbol for symbol in pd.DataFrame(self.wb_obj["ETF"].values)[1] if symbol not in [None, 'Symbol']]
        # Creating the dict
        return {"ETF": symbols}

    def forex_dict(self):
        # Getting the FOREX dict
        symbols = [symbol for symbol in pd.DataFrame(self.wb_obj["FOREX"].values)[1] if symbol not in [None, 'Symbol']]
        # Creating the dict
        return {"FOREX": symbols}

    def crypto_dict(self):
        # Getting the CRYPTO dict
        symbols = [symbol for symbol in pd.DataFrame(self.wb_obj["CRYPTO"].values)[1] if symbol not in [None, 'Symbol']]
        # Creating the dict
        return {"CRYPTO": symbols}
