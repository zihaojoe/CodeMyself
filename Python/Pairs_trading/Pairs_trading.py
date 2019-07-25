# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 12:06:03 2019

@author: Joker_C
"""
import tushare as ts
import numpy as np
import pandas as pd
import statsmodels
from statsmodels.tsa.stattools import coint
import seaborn   # 察协整关系
import matplotlib.pyplot as plt
from datetime import datetime
from pylab import *   # 动态设置字体，避免matplotlib里无法显示汉字字体
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

# tushare 文档1：https://blog.csdn.net/wolf1132/article/details/78606945
## tushare 官方文档：http://tushare.org/

# 配对交易 教程1：https://blog.csdn.net/CoderPai/article/details/80138774
# 配对交易 教程2：https://blog.csdn.net/hellocsz/article/details/82021684
## 协整(cointegration)和相关性(correlation)的区别

# 中国市场配对交易策略
class pairs_trading_cn():
    # 获取股票收盘价(close)面板
    def get_stock_panel(stockcode):
        '''
        Intro: 必须传递一个股票代码构成的list，返回其收盘价面板。（列为股票，行为时间）
        '''
        df = pd.DataFrame()
        for skd in stockcode:
           df[skd] = ts.get_k_data(skd)['close']
        df.fillna(method='ffill',inplace=True) # 停盘时用前面交易日的收盘价填充
        return df
    
    # 寻找整协的股票并绘制热度图
    def find_cointegrated_pairs(df):
        n = df.shape[1]   # 得到DataFrame长度
        pvalue_matrix = np.ones((n, n))   # 初始化p值矩阵
        df.columns = ts.get_industry_classified().set_index(['code']).loc[list(df.columns)]['name']
        keys = df.keys()   # 抽取列的名称
        pairs = [] # 初始化强协整组
        for i in range(n):
            for j in range(i+1, n):
                # 获取相应的两只股票的价格Series
                stock1 = df[keys[i]]
                stock2 = df[keys[j]]
                result = coint(stock1, stock2) # 分析它们的协整关系
                
                # 取出并记录p值
                pvalue = result[1]   # 第一个值是score，第二个值是pvalue
                pvalue_matrix[i, j] = pvalue
                if pvalue < 0.05:
                    pairs.append((keys[i], keys[j], pvalue))   # 记录股票对和相应的p值
        m = [0,0.2,0.4,0.6,0.8,1]
        seaborn.heatmap(pvalue_matrix, xticklabels=keys, 
                        yticklabels=keys, cmap='RdYlGn_r', 
                        mask = (pvalue_matrix >= 0.98))
        plt.show()
        return pvalue_matrix,pairs
    
    # 股票代码和股票名称的转换
    def transform(ls, industry=False):
        '''
        Intro: 传入一个list。如果传入的是name，则返回code；反之亦然。
        如果显式表明industry=True，则返回2维列表，第一个为name/code转换后的list，第二个为对应的行业/板块list。
        '''
        ls = list(ls)
        if industry == False:
            if (ls[0][0] >= '0') & (ls[0][0] <= '0'):
                return list(ts.get_industry_classified().set_index(['code']).loc[ls]['name'])   # ts.get_industry_classified()是获取所有股票的code/name/industry
            else:
                return list(ts.get_industry_classified().set_index(['name']).loc[ls]['code'])
        else:
            if (ls[0][0] >= '0') & (ls[0][0] <= '0'):
                return [list(ts.get_industry_classified().set_index(['code']).loc[ls]['name']),
                        list(ts.get_industry_classified().set_index(['code']).loc[ls]['c_name'])]
            else:
                return [list(ts.get_industry_classified().set_index(['name']).loc[ls]['code']),
                        list(ts.get_industry_classified().set_index(['name']).loc[ls]['c_name'])]
    
if __name__  == '__main__':
    # 输入股票列表
    stock_list = ['000333',
                  '000651',
                  '000418',
                  '600800',
                  '600576',
                  '600051']
    
    ts.get_industry_classified()   # 获取所有股票及行业分类    
    data = pairs_trading_cn.get_stock_panel(stock_list)   # 获取所选股票历史收盘价数据
    
    # 寻找整协的股票并绘制热度图
    pvalues, pair = pairs_trading_cn.find_cointegrated_pairs(data)
    print ('Stocks suitable for pairs trading are ', pair)
    
    selected_stock = pairs_trading_cn.get_stock_panel(pairs_trading_cn.transform(pair[0][0:2]))   # 获取适合配对交易的股票的收盘价数据
    
    # 画出所选两个股票的比值图
    S1 = selected_stock['600576']
    S2 = selected_stock['600051']
    ratios = S1 / S2
    ratios.plot()
    plt.axhline(ratios.mean())
    plt.legend([' Ratio'])
    plt.show()
    
    def zscore(series):
        return (series - series.mean()) / np.std(series)

    zscore(ratios).plot()
    plt.axhline(zscore(ratios).mean())
    plt.axhline(1.0, color='red')
    plt.axhline(-1.0, color='green')
    plt.show()


    
    
    
