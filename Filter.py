import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
import matplotlib.pyplot as plt
import yfinance as yf
import datetime
import seaborn as sns
from Topological_Embedding import Tembedding
from Funciones import varInfo, correlation_distance



class ETF_Filter:

    def __init__(self, etfs, tickers, start, end):

        self.start = start

        self.end = end

        self.tickers = sorted(tickers)
        
        self.etfs = sorted(etfs)
        
        self.df = yf.download(self.tickers + self.etfs, self.start, self.end)['Adj Close'].dropna()
        
        self.etf_filtration = self.df[self.etfs].copy()

        self.ticker_filtration = self.df[self.tickers].copy()

        self.pairs = []

        for i in range(len(self.etfs)):
            for j in range(len(self.tickers)):
                self.pairs.append((self.etfs[i],self.tickers[j]))
                
    def TopologicalEmbedding(self, tol=100, lagtime=1):
        
        
        topo = Tembedding(self.df)
        
        for pair in self.pairs.copy():
            resultado = topo.hausdorff(pair[0], pair[1])
            if resultado > tol:
                self.pairs.remove(pair)
            
        self.__filter_data()


    def correlationTest(self, tol = 0.9):
         

        for pair in self.pairs.copy():
                        
            resultado = np.corrcoef(self.etf_filtration[pair[0]],self.ticker_filtration[pair[1]])[0,1]
                
                
            if resultado < tol:

                self.pairs.remove(pair)
            
        self.__filter_data()



    def cointegration(self, tol=0.1):

        for pair in self.pairs.copy():

                resultado = coint(self.etf_filtration[pair[0]],self.ticker_filtration[pair[1]])[1]

                if resultado > tol:

                    self.pairs.remove(pair)
                    
        self.__filter_data()
        
    
    def distance_correlationTest(self, tol=0.3):
        
        for pair in self.pairs.copy():
            
            resultado = correlation_distance(self.etf_filtration[pair[0]],self.ticker_filtration[pair[1]])
            
            if resultado > tol:
                self.pairs.remove(pair)
                
        self.__filter_data()
   
                


    def variationalInformationTest(self, bins = 50, tol = 3.0):
    
        for pair in self.pairs.copy():

            resultado = varInfo(self.etf_filtration[pair[0]],self.ticker_filtration[pair[1]], bins = bins)

            if resultado > tol:

                self.pairs.remove(pair)
                
        self.__filter_data()

    
    def clear_filter(self):

        self.pairs = []
        self.etf_filtration = self.df[self.etfs].copy()
        self.ticker_filtration = self.df[self.tickers].copy()

        for i in range(len(self.etfs)):
            for j in range(len(self.tickers)):
                self.pairs.append((self.etfs[i],self.tickers[j]))
            
  
    def __filter_data(self):
    
        unique_tickers = []
        unique_etfs = []
        
        for pair in self.pairs:
            if pair[0] not in unique_etfs:
                unique_etfs.append(pair[0])
            if pair[1] not in unique_tickers:
                unique_tickers.append(pair[1])
        
        self.ticker_filtration = self.ticker_filtration[unique_tickers]
        self.etf_filtration = self.etf_filtration[unique_etfs]
        
        
class PairsFilter:

    def __init__(self, tickers, start, end):

        self.start = start

        self.end = end

        self.tickers = sorted(tickers)

        # Descargar los datos y guardarlos en un dataframe que se llame self.df

        self.df = yf.download(tickers, start, end)['Adj Close'].dropna()

        self.data_filtration = self.df.copy()

        self.pairs = []

        for i in range(len(self.tickers)):
            for j in range(i+1,len(self.tickers)):
                self.pairs.append((self.tickers[i],self.tickers[j]))
                
    def TopologicalEmbedding(self, tol=100, lagtime=1):
        
        
        topo = Tembedding(self.data_filtration)
        
        for pair in self.pairs.copy():
            resultado = topo.hausdorff(pair[0], pair[1])
            if resultado > tol:
                self.pairs.remove(pair)
            
        self.__filter_data()


    def correlationTest(self, tol = 0.9):
         

        for pair in self.pairs.copy():

                resultado = np.corrcoef(self.data_filtration[pair[0]],self.data_filtration[pair[1]])[0,1]
                if resultado < tol:

                    self.pairs.remove(pair)
            
        self.__filter_data()



    def cointegration(self, tol=0.1):

        for pair in self.pairs.copy():

                resultado = coint(self.data_filtration[pair[0]], self.data_filtration[pair[1]])[1]

                if resultado > tol:

                    self.pairs.remove(pair)
                    
        self.__filter_data()
        
    
    def distance_correlationTest(self, tol=0.3):
        
        for pair in self.pairs.copy():
            
            resultado = correlation_distance(self.data_filtration[pair[0]], self.data_filtration[pair[1]])
            
            if resultado > tol:
                self.pairs.remove(pair)
                
        self.__filter_data()
   
                


    def variationalInformationTest(self, bins = 50, tol = 3.0):
    
        for pair in self.pairs.copy():

            resultado = varInfo(self.data_filtration[pair[0]],self.data_filtration[pair[1]], bins = bins)

            if resultado > tol:

                self.pairs.remove(pair)
                
        self.__filter_data()

    
    def clear_filter(self):

        self.pairs = []
        self.data_filtration = self.df.copy()

        for i in range(len(self.tickers)):
            for j in range(i+1,len(self.tickers)):
                self.pairs.append((self.tickers[i],self.tickers[j]))
            
  
    def __filter_data(self):
    
        unique_tickers = []
        
        for pair in self.pairs:
            if pair[0] not in unique_tickers:
                unique_tickers.append(pair[0])
            if pair[1] not in unique_tickers:
                unique_tickers.append(pair[1])
        
        self.data_filtration = self.data_filtration[unique_tickers]

    def get_all_filters(self, tol_varinfo = 1.0, bins_varinfo = 30, tol_Temb = 80, tol_corr=0.8, tol_coint=0.1):
        self.correlationTest(tol = tol_corr)
        self.variationalInformationTest(tol = tol_varinfo, bins = bins_varinfo)
        self.TopologicalEmbedding(tol=tol_Temb)
        self.cointegration(tol=tol_coint)

        
    
    def plot_correlation(self, method = 'pearson', *args):
        plt.figure(figsize=(25,25))
        sns.heatmap(self.data_filtration.corr(method = method), cmap = sns.cm.rocket, *args)
        
    def plot_variational_information(self, bins= 50, *args):
        varinfo_matrix = np.zeros([self.data_filtration.shape[1], self.data_filtration.shape[1]])
        for i in range(self.data_filtration.shape[1]):
            for j in range(self.data_filtration.shape[1]):
                varinfo_matrix[i,j] = varInfo(self.data_filtration.iloc[:,i], self.data_filtration.iloc[:,j], bins = 50)
        varinfo_matrix = pd.DataFrame(varinfo_matrix, columns=self.data_filtration.columns)
        
        plt.figure(figsize=(25,25))
        sns.heatmap(varinfo_matrix, cmap = sns.cm.rocket, xticklabels=self.tickers, yticklabels=self.tickers, *args)
        
    def plot_correlationDistance(self, method = 'pearson', *args):
        
        cor = self.data_filtration.corr(method='pearson')
        metrica = np.sqrt(0.5*(1-cor))
        plt.figure(figsize=(25,25))
        sns.heatmap(metrica, cmap = sns.cm.rocket_r)
        
