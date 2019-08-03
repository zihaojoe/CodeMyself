# =========================================================
# 安装依赖库
# =========================================================
install.packages('ggplot2', dependencies = TRUE)
## ggplot参考文档 
## https://www.jianshu.com/p/0c25e1904026
install.packages('ggthemes', dependencies = TRUE)
install.packages('gridExtra', dependencies = TRUE)   # 一个画布上显示
install.packages('dplyr')   # 从来对数据进行分组统计
install.packages('alr3')   # 用来进行回归的数据集
install.packages('reshape2')   # 用来数据框数据透视
install.packages('GGally') # 用来绘制scatterplot matrix

# =========================================================
# lab1:绘频数图并进行level排序
# =========================================================
setwd(''C:/Users/01/Desktop/CodeMyself/R'')
library(ggplot2)
data = read.csv('reddit.csv')
str(data)   # str表示structure，对数据结构进行描述
levels(data$age.range)   # level表示等级，分类标签去重后的集合

qplot(data = data, x = age.range)   # 绘图

## 两种方法建立因子并排序（x轴排序）
data$age.range = ordered(data$age.range, levels = c('Under 18', '18-24', '25-34', '35-44', '45-54', '55-64', '65 of Above'))
data$age.range = factor(data$age.range, levels = c('Under 18', '18-24', '25-34', '35-44', '45-54', '55-64', '65 of Above'), order = T)
qplot(data = data, x = age.range)


# =========================================================
# lab2:单变量分析，facebook数据
# =========================================================
setwd('C:/Users/01/Desktop/CodeMyself/R')
library(ggthemes)
library(ggplot2)
pf = read.csv('pseudo_facebook.tsv', sep = '\t')

## 创建图并修改x轴的值
qplot(data = pf, x = dob_day)
qplot(data = pf, x = dob_day) + scale_x_continuous(breaks = 1:31)   #方法1
ggplot(aes(x = dob_day), data = pf) + geom_histogram(binwidth = 1) + scale_x_continuous(breaks = 1:31)   # 方法2

## 一页多图facet
qplot(data = pf, x = dob_day) + scale_x_continuous(breaks = 1:31) + facet_wrap(~dob_month, ncol = 4)
qplot(data = subset(pf, !is.na(gender)), x = dob_day) + scale_x_continuous(breaks = 1:31) + facet_wrap(~gender, ncol = 4)   #去除NA值,在原始数据中把gender的NA去掉
qplot(data = na.omit(pf), x = dob_day) + scale_x_continuous(breaks = 1:31) + facet_wrap(~gender, ncol = 4)   #去除NA值,在原始数据中把所有NA去掉

## 调整bin width（bar宽）
qplot(data = pf, x = friend_count) + scale_x_continuous(limits = c(0, 1000))   # x轴显示范围
qplot(data = pf, x = friend_count, binwidth = 25) + scale_x_continuous(limits = c(0, 1000)) + scale_x_continuous(limits = c(0, 1000), breaks = seq(0, 1000, 50))  # bin width

## 建立数据统计表
table(pf$gender)
by(pf$friend_count, pf$gender, summary)   #第一个参数是变量，第二个是分类依据，第三个是返回描述

## 一张画布多张图，坐标转换
library(gridExtra)
### 方法一：
p1 <- qplot(data = pf, x = friend_count)
p2 <- qplot(data = pf, x = log10(friend_count+1))
p3 <- qplot(data = pf, x = sqrt(friend_count+1))
grid.arrange(p1, p2, p3, ncol = 1)
### 方法二：
rm(p1);rm(p2);rm(p3)
p1 <- ggplot(data = pf, aes(x = friend_count)) + ggeom_histogram()
p2 <- p1 + scale_x_log10()
p3 <- p1 + scale_x_sqrt()
grid.arrange(p1, p2, p3, ncol = 1)

## 频数多边形(frequency polygon)
qplot(x = friend_count, y = ..count../sum(..count..),    #sum(..count..)是总样本量，如果要是每个自己比例，则用..density..
      data = subset(pf, !is.na(gender)), binwidth = 10,
      geom = "freqpoly", color = gender) +
  scale_x_continuous(limits  =  c(0, 1000), breaks = seq(0, 1000, 50))

## 盒型图(boxplot
qplot(data = subset(pf,!is.na(gender)), x = gender, y = friend_count, geom = 'boxplot')
qplot(data = subset(pf,!is.na(gender)), x = gender, y = friend_count, geom = 'boxplot', ylim = c(0,1000))   # 方法一
qplot(data = subset(pf,!is.na(gender)), x = gender, y = friend_count, geom = 'boxplot') + scale_y_continuous(limits = c(0,1000))   #方法二：注意方法一、二结果相同。但都是直接去除limit之外的点，所以和原图不一样

qplot(data = subset(pf,!is.na(gender)), x = gender, y = friend_count, geom = 'boxplot') + coord_cartesian(ylim = c(0,1000))   #方法三：和一、二不同，三只是调整坐标系显示范围

## 连续变量变成0-1分类变量(有很多0值时)
summary(pf$mobile_likes)
summary(pf$mobile_likes>0)
mobile_check_in <- NA   # 新变量初始化
pf$mobile_check_in <- ifelse(pf$mobile_likes>0, 1, 0)   
pf$mobile_check_in <- factor(pf$mobile_check_in)   # 转化为因子
sum(pf$mobile_check_in =  = 0)/length(pf$mobile_check_in)   # 计算每个分类的比例。注意factor变量不能用sum()


# =========================================================
# lab3:双变量分析，facebook数据
# =========================================================
setwd('C:/Users/01/Desktop/CodeMyself/R')
library(ggplot2)
pf  =  read.csv('pseudo_facebook.tsv', sep = '\t')

## 创建散点图
qplot(x = age, y = friend_count, data = pf)
qplot(age, friend_count, data = pf)   # 默认x轴在前面，y在后面
ggplot(aes(x = age, y = friend_count), data = pf) + geom_point() + xlim(13, 90)

## 降低重合度并添加噪声
ggplot(aes(x = age, y = friend_count), data = pf) + geom_point(alpha = 1/20) + xlim(13, 90)   # alpha表示20个重合时全黑
ggplot(aes(x = age, y = friend_count), data = pf) + geom_jitter(alpha = 1/20) + xlim(13, 90)   # age是不连续的，所以jitter添加扰动，使图看起来更连续

## 坐标值转换
ggplot(aes(x = age, y = friend_count), data = pf) + geom_point(alpha = 1/20) + xlim(13, 90) + coord_trans(y = 'sqrt')   # 注意不是y' = sqrt(y)，而是把坐标轴进行等比例刻度缩放
ggplot(aes(x = age, y = friend_count), data = pf) + geom_point(alpha = 1/20, position = position_jitter(height = 0)) + xlim(13, 90) + coord_trans(y = 'sqrt')   # 注意不是y' = sqrt(y)，而是把坐标轴进行等比例刻度缩放

## 分组统计
library(dplyr)
age_groups <- group_by(pf, age)
pf,fc_by_age <- summarise(age_groups,
                       friend_count_mean = mean(friend_count), 
                       friend_count_median = median(friend_count),
                       n = n())   # n = n()是用来统计每个分组的人数
pf,fc_by_age <- arrange(pf,fc_by_age, age)   # 排序

### 分组统计—管道操作符；
### %>%表示把前一个表达式的返回结果传入后一个表达式的参数
library(dplyr)
rm(pf,fc_by_age)
pf,fc_by_age <- pf %>% 
  group_by(age) %>%
  summarise(friend_count_mean = mean(friend_count),
            friend_count_median = median(friend_count),
            n = n())%>%    # n = n()是用来统计每个分组的人数
  arrange(age)   # 排序

ggplot(data = pf,fc_by_age, mapping = aes(x = age, y = friend_count_mean))+geom_bar(stat = 'identity')   #如果条形图中的y值已知，要stat改成'identity'，否则默认是'count'，即观测数量

## 把原数据散点图和平均值折线图放在一张图中
ggplot(aes(x = age, y = friend_count), data = pf) 
  + geom_point(alpha = 1/20, position = position_jitter(height = 0), color = 'orange') 
  + xlim(13, 90) + coord_trans(y = 'sqrt') 
  + geom_line(stat = 'summary', fun.y = mean)   #对于fun中需要的其他参数，则用fun.args = 来表示，例如：fun.y = quantile, fun.args = list(probs = 0.9).

## 相关系数Correlation coefficient
cor.test(pf$age, pf$friend_count, method = 'pearson')
with(pf, cor.test(age, friend_count, method = 'pearson'))   # with语句把所有的操作限制在数据框中

### 绘制回归线
ggplot(data = pf, aes(x = pf$www_likes_received, y = pf$likes_received)) + geom_point() +
  xlim(0, quantile(pf$www_likes_received, 0.95)) +   # 显示95%的x轴数据
  ylim(0, quantile(pf$likes_received, 0.95)) +
  geom_smooth(method = 'lm', color = 'red')   # 绘制回归线

## 回归案例
library(alr3)
data(Mitchell)
qplot(data = Mitchell, Month, Temp)   # 图看起来correlation是0
cor.test(Mitchell$Month, Mitchell$Temp, method = 'pearson')   # 但实际上是有一定相关性的
ggplot(aes(x = (Month%%12),y = Temp), data = Mitchell) + geom_point()

### 增加和减少图形密度
p1 <- ggplot(data = pf, aes(x = age, y = friend_count)) + geom_line(stat = 'summary', fun.y = mean)   #原图
p2 <- ggplot(data = pf, aes(x = round(age/5)*5, y = friend_count)) + geom_line(stat = 'summary', fun.y = mean)   #以5为间距合并，缩小密度
p3 <- ggplot(data = pf, aes(x = age, y = friend_count)) + geom_line(stat = 'summary', fun.y = mean) + geom_smooth()  #原图+自动smooth
grid.arrange(p1, p2, p3, ncol = 1)


# =========================================================
# lab4-1:多变量分析，facebook数据
# =========================================================
setwd('C:/Users/01/Desktop/CodeMyself/R')
library(ggplot2)
library(dplyr)
pf  =  read.csv('pseudo_facebook.tsv', sep = '\t')

## 分组统计
ggplot(data = subset(pf, !is.na(gender)), aes(x = gender, y = age)) + geom_boxplot() +
  stat_summary(fun.y = mean, geom = 'point', shape = 3)   #分组并加入统计变量
ggplot(data = subset(pf, !is.na(gender)), aes(x = age, y = friend_count)) +
  geom_line(aes(color = gender),stat = 'summary', fun.y = median) + scale_color_manual(values = c('red','green'))   #scale_color_manual/scale_fill_manual自定义颜色

###先分组统计，再绘制男女折线图
age_gender_groups <- group_by(pf, age, gender)   #也可用管道操作符操作
age_gender_groups <- filter(age_gender_groups, !is.na(gender))
pf.fc_by_age_gender <- summarise(age_gender_groups,
                                mean_friend_count = mean(friend_count),
                                median_friend_count = median(friend_count),
                                n = n())
pf.fc_by_age_gender <- arrange(ungroup(pf.fc_by_age_gender), age, gender)   #summarise只会解绑一维，得到的数据框仍是分组的，所以要再解绑

ggplot(data = pf.fc_by_age_gender, aes(x = age, y = median_friend_count)) + geom_line(aes(color = gender))

### 数据透视：数据框长数据和宽数据格式转换（long/wide cast和melt，melt见后文）
library(reshape2)
pf.fc_by_age_gender   # 转换前—长数据
pf.fc_by_age_gender.wide <- dcast(pf.fc_by_age_gender,    # dcast表示返回dataframe，如果是array则是acast
                                  age ~ gender,   # ~之前是保持不变的变量，后面是需要分列的变量
                                  value.var = 'median_friend_count')   # 转换后-宽数据

### 男女比例折线图
ggplot(data = pf.fc_by_age_gender.wide, aes(x = age, y = female/male,)) + 
  geom_line() +
  geom_hline(yintercept = 1, alpha = 0.4, linetype = 2)   # horizontal line, alpha是调整线透明度

## 多个数值变量分析，cut函数分组
### age和friend_count分组图
pf$year_joined <- floor(2014 - pf$tenure/365)
table(pf$year_joined)
pf$year_joined.bucket <- cut(pf$year_joined, c(2004, 2009, 2011, 2012, 2014))   #用cut分组，2004和2014分别是头和尾
ggplot(data = subset(pf, !is.na(year_joined.bucket)), aes(x = age, y = friend_count)) +
  geom_line(aes(color = year_joined.bucket),stat = 'summary', fun.y = mean) + 
  geom_line(stat = 'summary', fun.y = mean, linetype = 2)   # 加入总体的均值

### tenure和friendships_initiated分组图
ggplot(data = subset(pf, tenure >= 1), aes(x = tenure, y = friendships_initiated / tenure)) +
  geom_line(aes(color = year_joined.bucket),stat = 'summary', fun.y = mean)

ggplot(data = subset(pf, tenure >= 1), aes(x = round(tenure/7) * 7, y = friendships_initiated / tenure)) +
  geom_line(aes(color = year_joined.bucket),stat = 'summary', fun.y = mean) 

ggplot(data = subset(pf, tenure >= 1), aes(x = tenure, y = friendships_initiated / tenure)) +
  geom_smooth(aes(color = year_joined.bucket))


# =========================================================
# lab4-2:多变量分析，酸奶数据
# =========================================================
setwd('C:/Users/01/Desktop/CodeMyself/R')
library(ggplot2)
yo  =  read.csv('yogurt.csv')
str(yo)
yo$id <- factor(yo$id)

ggplot(data = yo, aes(x = price)) + geom_histogram(binwidth = 10, fill = 'orange')

## 新增一列合计
yo <- transform(yo, all.perchase = strawberry + blueberry + pina.colada + plain + mixed.berry)   # 方法一
yo$all.perchase <- yo$strawberry + yo$blueberry + yo$pina.colada + yo$plain + yo$mixed.berry   # 方法二

## 价格-时间散点图
ggplot(data = yo, aes(x = time, y = price)) + geom_jitter(alpha = 1/4, shape = 21)

## 随机抽取16个id进行进一步考察
set.seed(4230)
sample.ids <- sample(levels(yo$id), 16)

ggplot(aes(x = time, y = price), data = subset(yo, id %in% sample.ids)) +   # %in% 类似于python isin()
  facet_wrap(~id) + geom_line() + geom_point(aes(size = all.perchase), pch = 5)   # pch和shape一样，用来表示点的形状

## 创建散点图矩阵
library(GGally)
theme_set(theme_minimal(20))
set.seed(1836)
pf_subset <- pf[,c(10:15)]
ggpairs(pf_subset[sample.int(nrow(pf_subset),10),])

## 基因数据热度图
nci <- read.table('nci.tsv')
colnames(nci) <- c(1:64)   # 修改列名

library(reshape2)
nci.long.samp <- melt(as.matrix(nci[1:200, ]))
names(nci.long.samp) <- c('gene','case','value')

ggplot(aes(y = gene, x = case, fill = value),
       data = nci.long.samp) +
  geom_tile()+
  scale_fill_gradientn(colors = colorRampPalette(c('blue', 'red'))(100))

# =========================================================
# lab5:多变量分析，酸奶数据
# =========================================================
library(ggplot2)
data(diamonds)   # 加载ggplot中的diamonds数据集

ggplot(aes(x = carat, y = price), data = diamonds) +
  scale_x_continuous(lim = c(0, quantile(diamonds$carat, 0.99))) +
  scale_y_continuous(lim = c(0, quantile(diamonds$price, 0.99))) +
  geom_point(fill = I('#f97420'), color = I('black'), shape = 21)   # I表示AsIs，一种数据格式，是什么样就是什么样

## 安装相应包
install.packages('scales')
install.packages('memisc')
install.packages('lattice')
install.packages('MASS')
install.packages('car')
library(ggplot2)
library(GGally)
library(scales)
library(memisc)

set.seed(20022012)
diamond_samp <- diamonds[sample(1:length(diamonds$price), 10000), ]
ggpairs(diamond_samp, 
        lower = list(continuous = wrap("points", shape = I('.'))),
        upper = list(combo = wrap("box", outlier.shape = I('.'))))

## 坐标转换
ggplot(aes(x = carat, y = price), data = diamonds) +
  scale_y_continuous(trans = log10_trans()) +
  geom_point(fill = I('#f97420'), color = I('black'), shape = 21)

cuberoot_trans = function() trans_new('cuberoot',
                                     transform = function(x) x^(1/3),
                                     inverse = function(x) x^3)   # 返回立方根的函数

## 加入纯度作为颜色，坐标进行三次根号转换，添加透明度
ggplot(aes(carat, price, color = clarity), data = diamonds) +   geom_point(alpha = 0.2, size = 0.75, position = 'jitter') + 
  scale_color_brewer(type = 'div',
                     guide = guide_legend(title = 'Clarity', reverse = T,
                                          override.aes = list(alpha = 1, size = 2))) +
  scale_x_continuous(trans = cuberoot_trans(), limits = c(0.2, 3), breaks = c(0.2, 0.5, 1, 2, 3)) + 
  scale_y_continuous(trans = log10_trans(), limits = c(350, 15000), breaks = c(350, 1000, 5000, 10000, 15000)) +
  ggtitle('Price (log10) by Cube-Root of Carat')

## 回归
m1 <- lm(I(log(price)) ~ I(carat^(1/3)), data = diamonds)   # I函数表示把值转换后再进行回归，而不是把计算公式当成回归模型的一部分
m2 <- update(m1, ~ . + carat)   # 添加更多变量
m3 <- update(m2, ~ . + cut)
m4 <- update(m3, ~ . + color)
m5 <- update(m4, ~ . + clarity)
mtable(m1, m2, m3, m4, m5, sdigits = 3)   # 设置小数点


