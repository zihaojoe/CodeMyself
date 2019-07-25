# ----------------------------------------------------
# 安装依赖库
install.packages('ggplot2', dependencies = TRUE)
install.packages('ggthemes', dependencies = TRUE)

# ----------------------------------------------------
# lab1:绘图并进行level排序
setwd('C:/Users/01/Downloads')
library(ggplot2)
data = read.csv('reddit.csv')
str(data)   # str表示structure，对数据结构进行描述
levels(data$age.range)

qplot(data=data, x=age.range)   # 绘图

## 建立因子
data$age.range = ordered(data$age.range, levels=c('Under 18', '18-24', '25-34', '35-44', '45-54', '55-64', '65 of Above'))
data$age.range = factor(data$age.range, levels=c('Under 18', '18-24', '25-34', '35-44', '45-54', '55-64', '65 of Above'), order=T)
qplot(data=data, x=age.range)

# ----------------------------------------------------
# lab2:Facebook虚假数据统计分析
setwd('C:/Users/01/Downloads')
library(ggthemes)
pf = read.csv('pseudo_facebook.tsv', sep='\t')

## 创建图并修改x轴的值
qplot(data=pf, x=dob_day)
qplot(data=pf, x=dob_day) + scale_x_continuous(breaks=1:31)   #方法1
ggplot(aes(x = dob_day), data = pf) + geom_histogram(binwidth = 1) + scale_x_continuous(breaks = 1:31)   # 方法2

## 一页多图facet
qplot(data=pf, x=dob_day) + scale_x_continuous(breaks=1:31) + facet_wrap(~dob_month, ncol=4)

## 调整bin width（bar宽）
qplot(data=pf, x=friend_count) + scale_x_continuous(limits=c(0,1000) )





