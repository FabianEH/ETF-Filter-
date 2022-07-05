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
    def __init__(self, ticker, start, end):
        self.tseries = []
        self.tseries = yf.download(ticker, start = start, end = end)['Adj Close']
        self.tseries = self.tseries.dropna()
        self.tseries = self.tseries.reset_index(drop=True)
    def hausdorff(self, ticker_1, ticker_2, lagtime = 1):
        self.u = []
        self.v = []
        self.hausdorff_XY = []
        self.hausdorff_YX = []
        self.output = []
        self.u = self.tseries[ticker_1]
        self.v = self.tseries[ticker_2]
        self.u = np.transpose( np.array([  self.u[range(0, len(self.u) - lagtime, 1)], self.u[range(lagtime, len(self.u), 1)] ] ) )
        self.v = np.transpose( np.array([  self.v[range(0, len(self.v) - lagtime, 1)], self.v[range(lagtime, len(self.v), 1)] ] ) )
        self.hausdorff_XY = directed_hausdorff(self.u, self.v)[0]
        self.hausdorff_YX = directed_hausdorff(self.v, self.u)[0]
        self.output = max(self.hausdorff_XY, self.hausdorff_YX)    
    def graph(self, t_embedding):
        self.Xticker = []
        self.Yticker = []
        self.Xticker = t_embedding[:,0]
        self.Yticker = t_embedding[:,1]
        self.color_2d = np.linspace(0,1, num=len(self.Xticker))
        plt.scatter(self.Xticker, self.Yticker, c = self.color_2d)
        plt.show()
    def graph3d(self, ticker_embedding):
        self.X3d = []
        self.X3d = ticker_embedding[:,0]
        self.Y3d = []
        self.Y3d = ticker_embedding[:,1]
        self.Z3d = []
        self.Z3d = np.arange(len(self.X3d))
        self.pdticker = pd.DataFrame()
        self.pdticker['x'] = self.X3d
        self.pdticker['y'] = self.Y3d
        self.pdticker['z'] = self.Z3d
        fig = px.scatter_3d(self.pdticker, x='x', y='y', z='z', color = self.pdticker['z'])
        fig.show()

Topo = Tembedding(['AAPL', 'KR', 'JNJ', 'ICE', 'TGT', 'BABA', 'LUNA', 'BTC', 'EURUSD=X'], '2018-01-01', '2022-06-02') #Cargamos Ticker en los periodos descritos 
Topo.hausdorff('AAPL', 'KR', 3) #Seleccionamos los dos ticker que haremos topo-embedding y el lagtime
Topo.output #Resultado de Hausdorff
Topo.graph(Topo.u) #Grafica el topo-embedding del primer ticker 2D
Topo.graph3d(Topo.u) #Grafica el topo-embedding del primer ticker 3D


'''
Topo.hausdorff_XY #Resultado de Hausdorff (X, Y)
Topo.hausdorff_YX #Resultado de Hausdorff (Y, X)
Topo.u #Topo-embedding del primer ticker
Topo.v #Topo-embedding del segundo ticker
Topo.graph(Topo.u) #Grafica el topo-embedding del primer ticker 2D
Topo.graph(Topo.v) #Grafica el topo-embedding del segundo ticker 2D
Topo.graph3D(Topo.u) #Grafica el topo-embedding del primer ticker 3D
Topo.graph3D(Topo.v) #Grafica el topo-embedding del primer ticker 3D
'''