# ====================================================
# 安装依赖库
# ====================================================
install.packages('ggplot2', dependencies = TRUE)
install.packages('ggthemes', dependencies = TRUE)
install.packages('gridExtra', dependencies = TRUE)   # 一个画布上显示
install.packages('dplyr')   # 从来对数据进行分组统计
## ggplot参考文档 
## https://www.jianshu.com/p/0c25e1904026

# ====================================================
# lab1:绘频数图并进行level排序
# ====================================================
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

# ====================================================
# lab2:单变量分析
# ====================================================
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

## 一张画布多张图，坐标值转换
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
qplot(x=friend_count, y=..count../sum(..count..),    #sum(..count..)是总样本量，如果要是每个自己比例，则用..density..
      data=subset(pf, !is.na(gender)), binwidth=10,
      geom="freqpoly", color=gender) +
  scale_x_continuous(limits = c(0, 1000), breaks=seq(0, 1000, 50))

## 盒型图(boxplot
qplot(data=subset(pf,!is.na(gender)), x=gender, y=friend_count, geom='boxplot')
qplot(data=subset(pf,!is.na(gender)), x=gender, y=friend_count, geom='boxplot', ylim=c(0,1000))   # 方法一
qplot(data=subset(pf,!is.na(gender)), x=gender, y=friend_count, geom='boxplot') + scale_y_continuous(limits=c(0,1000))   #方法二：注意方法一、二结果相同。但都是直接去除limit之外的点，所以和原图不一样

qplot(data=subset(pf,!is.na(gender)), x=gender, y=friend_count, geom='boxplot') + coord_cartesian(ylim=c(0,1000))   #方法三：和一、二不同，三只是调整坐标系显示范围

## 连续变量变成0-1分类变量(有很多0值时)
summary(pf$mobile_likes)
summary(pf$mobile_likes>0)
mobile_check_in <- NA   # 新变量初始化
pf$mobile_check_in <- ifelse(pf$mobile_likes>0, 1, 0)   
pf$mobile_check_in <- factor(pf$mobile_check_in)   # 转化为因子
sum(pf$mobile_check_in==0)/length(pf$mobile_check_in)   # 计算每个分类的比例。注意factor变量不能用sum()

# ====================================================
# lab3:双变量分析
# ====================================================
setwd('C:/Users/01/Downloads')
library(ggthemes)
library(ggplot2)
pf = read.csv('pseudo_facebook.tsv', sep='\t')

## 创建散点图
qplot(x=age, y=friend_count, data=pf)
qplot(age, friend_count, data=pf)   # 默认x轴在前面，y在后面
ggplot(aes(x=age, y=friend_count), data=pf) + geom_point() + xlim(13, 90)

## 降低重合度并添加噪声
ggplot(aes(x=age, y=friend_count), data=pf) + geom_point(alpha=1/20) + xlim(13, 90)   # alpha表示20个重合时全黑
ggplot(aes(x=age, y=friend_count), data=pf) + geom_jitter(alpha=1/20) + xlim(13, 90)   # age是不连续的，所以jitter添加扰动，使图看起来更连续

## 坐标值转换
ggplot(aes(x=age, y=friend_count), data=pf) + geom_point(alpha=1/20) + xlim(13, 90) + coord_trans(y='sqrt')   # y'=sqrt(y)
ggplot(aes(x=age, y=friend_count), data=pf) + geom_point(alpha=1/20, position=position_jitter(height=0)) + xlim(13, 90) + coord_trans(y='sqrt')   # y'=sqrt(y)

## 分组统计
library(dplyr)
age_groups <- group_by(pf, age)
fc_by_age <- summarise(age_groups,
                       friend_count_mean=mean(friend_count), 
                       friend_count_median=median(friend_count),
                       n=n())   # n=n()是用来统计每个分组的人数
fc_by_age <- arrange(fc_by_age, age)   # 排序

### 分组统计—管道操作符；
### %>%把前一个表达式的返回结果传入后一个表达式的参数
library(dplyr)
rm(fc_by_age)
fc_by_age <- pf %>% 
  group_by(age) %>%
  summarise(friend_count_mean=mean(friend_count),
            friend_count_median=median(friend_count),
            n=n())%>%    # n=n()是用来统计每个分组的人数
  arrange(age)   # 排序

ggplot(data=fc_by_age, mapping=aes(x=age, y=friend_count_mean))+geom_bar(stat='identity')   #如果条形图中的y值已知，要stat改成'identity'，否则默认是'count'，即观测数量

##把原数据散点图和平均值折线图放在一张图中
ggplot(aes(x=age, y=friend_count), data=pf) 
  + geom_point(alpha=1/20, position=position_jitter(height=0), color='orange') 
  + xlim(13, 90) + coord_trans(y='sqrt') 
  + geom_line(stat='summary', fun.y=mean)   #对于fun中需要的其他参数，则用fun.args=来表示，例如：fun.y=quantile, fun.args=list(probs=0.9).



