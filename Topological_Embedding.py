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
        self.tseries = []
        self.tseries = df
        self.tseries = self.tseries.dropna()
        self.tseries = self.tseries.reset_index(drop=True)
    def hausdorff(self, ticker_1, ticker_2, lagtime = 1):
        self.u = self.tseries[ticker_1]
        self.v = self.tseries[ticker_2]
        self.u = np.transpose( np.array([  self.u[range(0, len(self.u) - lagtime, 1)], self.u[range(lagtime, len(self.u), 1)] ] ) )
        self.v = np.transpose( np.array([  self.v[range(0, len(self.v) - lagtime, 1)], self.v[range(lagtime, len(self.v), 1)] ] ) )
        self.hausdorff_XY = directed_hausdorff(self.u, self.v)[0]
        self.hausdorff_YX = directed_hausdorff(self.v, self.u)[0]
        self.output = max(self.hausdorff_XY, self.hausdorff_YX)
        return max(self.hausdorff_XY, self.hausdorff_YX)     
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
