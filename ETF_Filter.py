import pandas as pd
import numpy as np
import pandas_datareader.data as data_web
import datetime as dt
import matplotlib.pyplot as plt
import plotly.express as px
import scipy
import scipy.spatial
from scipy.spatial.distance import directed_hausdorff
import yfinance as yf
from numpy.fft import fft, ifft

class Tembedding: 
    def __init__(self, df):
        self.tseries = df
        
    def tset(self, ticker, lagtime):
        self.ticker_tset = []
        for tao in (range(int(self.tseries.shape[0]/lagtime)//1)):
            a = self.tseries[(lagtime*tao)]
            self.ticker_tset.append(a)
    def tembedding(self):
        self.ticker_embedding = []
        for i in range(len(self.ticker_tset)-1):
            point = (self.ticker_tset[i], self.ticker_tset[i+1])
            self.ticker_embedding.append(point)   
    def metric(self, tickers, str_tickers):
        self.ticker_hausdorff = pd.DataFrame()
        self.ticker_hausdorff.index = str_tickers
        measures = []
        for i in tickers:
                a = directed_hausdorff(np.array(self.ticker_embedding), np.array(i))[0]
                measures.append(a)
        self.ticker_hausdorff['Distances'] = measures 
    
    def hausdorff_metric(self, X,Y):
        return min(directed_hausdorff(X,Y)[0], directed_hausdorff(Y,X)[0])
    def graph(self, t_embedding):
        self.Xticker = []
        for i in range(len(t_embedding)):
            x_te = t_embedding[i][0]
            self.Xticker.append(x_te)
        self.Yticker = []
        for i in range(len(t_embedding)):
            y_te = t_embedding[i][1]
            self.Yticker.append(y_te)
        color = np.linspace(0,1, num=len(self.Xticker))
        plt.scatter(self.Xticker,self.Yticker, c = color)
        plt.show()
    def tembedding3d(self, ticker_tset):
        self.ticker_embedding3d = []
        for i in range(len(ticker_tset)-1):
            point = (ticker_tset[i], ticker_tset[i+1], i)
            self.ticker_embedding3d.append(point)
    def graph3d(self, ticker_embedding):
        self.X3d = []
        for i in range(len(ticker_embedding)):
            x_te = ticker_embedding[i][0]
            self.X3d.append(x_te)
        self.Y3d = []
        for i in range(len(ticker_embedding)):
            y_te = ticker_embedding[i][1]
            self.Y3d.append(y_te)
        self.Z3d = []
        for i in range(len(ticker_embedding)):
            z_te = ticker_embedding[i][2]
            self.Z3d.append(z_te)
        self.pdticker = pd.DataFrame()
        self.pdticker['x'] = self.X3d
        self.pdticker['y'] = self.Y3d
        self.pdticker['z'] = self.Z3d
        fig = px.scatter_3d(self.pdticker, x='x', y='y', z='z', color = self.pdticker['z'])
        fig.show()