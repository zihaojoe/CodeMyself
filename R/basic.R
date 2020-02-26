# =========================================================
# Read and write Files
# =========================================================
### read.csv()
gapminder <- read.csv("data/gapminder5.csv", stringsAsFactors=FALSE)
gapminder <- read.csv(file = "data/gapminder5.csv",
                      sep = ",",
                      stringsAsFactors = FALSE)
library(readr)
gapminder <- read_csv("data/gapminder5.csv")   # read_csv is from the 'readr'

### install.packages("here")
### here() gets your current working directory and appends any strings enclosed in the function
here::here()

### write.csv()
write.csv(gapminder07, file = "data/gapminder07.csv", row.names = FALSE)
# =========================================================
# Vector
# =========================================================
example <- c(7,8,9)
example[2]
example[example>7]

set.seed(1234) 
x1 <- rnorm(5)
x2 <- rnorm(20, mean=0.5)
x1[3]
x3 <- x2[1:5]
x1[-3]

### miss values
vec <- c(1, 8, NA, 7, 3)
mean(vec)
mean(vec, na.rm=TRUE)

# =========================================================
# Matrix
# =========================================================
mat <- matrix(data=c(1,2,3,4,5,6,11,12,34), ncol=3)
mat[1,]
mat[1,3]

# =========================================================
# Data frames
# =========================================================
df <- data.frame(candidate=c("Biden","Warren","Sanders"), 
                 poll=c(26,17,17), 
                 age=c(76,70,78))
df[1,3]
df$age
df[['age']]
df$age[df$candidate=="Biden"]
names(df)   # list all the columns
dim(df)   # dimension
str(df)    # all columns and datatype. Structure of the Dataframe
nrow(df)   # number of rows
mean(df$age)   # mean(), median(), var(), sd(), quantile()

### add new variable
gapminder$newvar <- newvar
gapminder <- cbind(gapminder, newvar)
### removing
gapminder$newvar <- NULL
gapminder <- gapminder[-"newvar"]

gapminder <- read.csv("data/gapminder5.csv", stringsAsFactors=FALSE)
summary(gapminder)   # summary
table(gapminder$continent)   # frequency table of the variable
prop.table(table(gapminder$continent))   # proportion table

### sort(), order(), and rank()
gapminder07 <- subset(gapminder, subset = year==2007)
sort(table(gapminder07$continent))
gapminder07$pop[gapminder07$country=="Mexico"]
head(gapminder07[order(gapminder07$pop, decreasing=TRUE),])   # order后要加,

### recoding
gapminder07$lifeExp_round <- round(gapminder07$lifeExp)
gapminder07$lifeExp_highlow[gapminder07$lifeExp>mean(gapminder07$lifeExp)] <- "High"
gapminder07$lifeExp_highlow[gapminder07$lifeExp<mean(gapminder07$lifeExp)] <- "Low"
table(gapminder07$lifeExp_highlow)

### aggregating, group
### aggregate(y ~ x, FUN = mean) gives the mean of vector y for each unique group in x
aggregate(gapminder07$lifeExp ~ gapminder07$continent, FUN = mean)
aggregate(lifeExp ~ continent, data = gapminder07, FUN = mean)

### statistics
cor(gapminder07$lifeExp, gapminder07$gdpPercap)   # correlatioin
t1 <- t.test(gapminder07$gdpPercap~gapminder07$lifeExp_highlow)   # ttest
t1 <- t.test(gdpPercap~lifeExp_highlow, data=gapminder07)
reg1 <- lm(lifeExp ~ gdpPercap + pop, data = gapminder07)   # linear regression
summary(reg1)

# =========================================================
# Visualization & plots
# =========================================================
### histgram
hist(gapminder07$lifeExp, 
     main="Distribution of life expectancy across countries in 2007", 
     xlab="Life expectancy", ylab="Frequency")

### scatterplot (y ~ x)
plot(gapminder07$lifeExp ~ gapminder07$gdpPercap,
     main="Relationship between life expectancy and GDP per capita in 2007", 
     ylab="Life expectancy", xlab="GDP per capita")

### abline
abline(h = mean(gapminder07$lifeExp))