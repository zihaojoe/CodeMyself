# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 15:02:33 2019

@author: Joe Cheung
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import xlrd
import math
from pylab import *
from sklearn.linear_model import LinearRegression
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

input_filename = r'.\Example.xlsx'
output_filename = r'.\Example.jpeg'

# 线性回归，返回拟合后的x, y
def linearfit(x,y):
    x = x.reshape(-1,1)
    model =  LinearRegression()
    model.fit(x, y)
    y2 = model.predict(x)
    return x, y2

workbook = xlrd.open_workbook(input_filename)
sheet_names = workbook.sheet_names()

#建立画布和图层
fig = plt.figure(num=1, figsize=(20, 10), dpi=60)
ax = plt.subplot(111)   # 231表示原画布分成2行3列，子图在第1个位置（从上往下，从左往右标号）
fig.add_subplot(ax)

createVar = globals()   # 创建全局变量保存图名，可以动态生成一系列变量
for time, i in enumerate(sheet_names):
    df = pd.read_excel(input_filename, sheet_name=i, index_col=None)
    createVar['p' + str(time)], = ax.plot(df['时间'].astype(str), df['成交金额'], linewidth=1.5)   # 动态生成变量并绘制原折线图， p, = 表示解构单元素列表
    x, y = linearfit(np.arange(len(df['成交金额'])), df['成交金额'])   
    ax.plot(x, y, linewidth=1.5, color='gray', linestyle='--')   # 绘制回归图

# 设置图片部分属性
plt.title('成交量追踪图', fontsize=30)
plt.xlabel('日期', fontsize=20, verticalalignment='top')   # verticalalignment表示xlabel的那个文本用哪个部位去对其相应的位置
plt.ylabel('成交金额', fontsize=20)
## 设置坐标刻度
locs, labels = xticks()   # 获取x坐标刻度的位置和对应的文本值
ax.xaxis.set_major_locator(ticker.MultipleLocator(5))   #5 表示每隔5个进行一次刻度，值越大，刻度密度越小
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
## 使得x坐标刻度旋转
for tick in ax.get_xticklabels():   
    tick.set_rotation(60)
## 添加图例
ax.legend([p0, p1], sheet_names, loc = 'upper left', ncol=5, fontsize=20)   # 第一个列表是图，第二个是图例名称，ncol是图例列的个数

fig.savefig(output_filename, dpi=300, bbox_inches = 'tight')   #bbox_inches参数为了解决图片显示不完整
