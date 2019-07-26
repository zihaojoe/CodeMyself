# ----------------------------------------------------
# 安装依赖库
install.packages('ggplot2', dependencies = TRUE)
install.packages('ggthemes', dependencies = TRUE)
install.packages('gridExtra', dependencies = TRUE)   # 一个画布上显示
## ggplot参考文档 
## https://www.jianshu.com/p/0c25e1904026

# ----------------------------------------------------
# lab1:绘图并进行level排序
setwd('C:/Users/01/Downloads')
library(ggplot2)
data = read.csv('reddit.csv')
str(data)   # str表示structure，对数据结构进行描述
levels(data$age.range)   # level表示等级，分类标签去重后的集合

qplot(data=data, x=age.range)   # 绘图

## 两种方法建立因子并排序（x轴排序）
data$age.range = ordered(data$age.range, levels=c('Under 18', '18-24', '25-34', '35-44', '45-54', '55-64', '65 of Above'))
data$age.range = factor(data$age.range, levels=c('Under 18', '18-24', '25-34', '35-44', '45-54', '55-64', '65 of Above'), order=T)
qplot(data=data, x=age.range)

# ----------------------------------------------------
# lab2:Facebook虚假数据统计分析
setwd('C:/Users/01/Downloads')
library(ggthemes)
library(ggplot2)
pf = read.csv('pseudo_facebook.tsv', sep='\t')

## 创建图并修改x轴的值
qplot(data=pf, x=dob_day)
qplot(data=pf, x=dob_day) + scale_x_continuous(breaks=1:31)   #方法1
ggplot(aes(x = dob_day), data = pf) + geom_histogram(binwidth = 1) + scale_x_continuous(breaks = 1:31)   # 方法2

## 一页多图facet
qplot(data=pf, x=dob_day) + scale_x_continuous(breaks=1:31) + facet_wrap(~dob_month, ncol=4)
qplot(data=subset(pf, !is.na(gender)), x=dob_day) + scale_x_continuous(breaks=1:31) + facet_wrap(~gender, ncol=4)   #去除NA值,在原始数据中把gender的NA去掉
qplot(data=na.omit(pf), x=dob_day) + scale_x_continuous(breaks=1:31) + facet_wrap(~gender, ncol=4)   #去除NA值,在原始数据中把所有NA去掉

## 调整bin width（bar宽）
qplot(data=pf, x=friend_count) + scale_x_continuous(limits=c(0, 1000))   # x轴显示范围
qplot(data=pf, x=friend_count, binwidth=25) + scale_x_continuous(limits=c(0, 1000)) + scale_x_continuous(limits=c(0, 1000), breaks=seq(0, 1000, 50))  # bin width

## 建立数据统计表
table(pf$gender)
by(pf$friend_count, pf$gender, summary)   #第一个参数是变量，第二个是分类依据，第三个是返回描述

## 一张画布多张图
library(gridExtra)
### 方法一：
p1 <- qplot(data=pf, x=friend_count)
p2 <- qplot(data=pf, x=log10(friend_count+1))
p3 <- qplot(data=pf, x=sqrt(friend_count+1))
grid.arrange(p1, p2, p3, ncol=1)
### 方法二：
rm(p1);rm(p2);rm(p3)
p1 <- ggplot(data=pf, aes(x=friend_count)) + ggeom_histogram()
p2 <- p1 + scale_x_log10()
p3 <- p1 + scale_x_sqrt()
grid.arrange(p1, p2, p3, ncol=1)

## 频数多边形(frequency polygon)
qplot(x = friend_count, y = ..count../sum(..count..), 
      data = subset(pf, !is.na(gender)), binwidth=10,
      geom="freqpoly", color=gender) +
  scale_x_continuous(limits = c(0, 1000), breaks = seq(0, 1000, 50))

