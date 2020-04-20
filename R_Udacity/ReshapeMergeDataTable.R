# =========================================================
# Reshape2
# =========================================================
library(here)
library(readr)
generation <- read.csv(here::here("R/data/ca_energy_generation.csv"), stringsAsFactors=F)
imports <- read.csv(here::here("R/data/ca_energy_imports.csv"), stringsAsFactors=F)
str(generation)   # structure
class(generation$datetime)   # show the class/datatype
library(lubridate)
generation$datetime <- as_datetime(generation$datetime)
imports$datetime <- as_datetime(imports$datetime)

### melt -> make data long
### dcast -> make data wide
### recast ->  melt then cast data
library(reshape2)
long_gen <- melt(generation, id.vars = "datetime",
                 variable.name = "source",
                 value.name = "usage")
head(long_gen)
head(long_gen[order(long_gen$datetime), ])

long_merged_energy <- melt(merged_energy, id.vars = "datetime",
                           variable.name = "source",
                           value.name = "usage")
head(long_merged_energy)

### Merging data
# merge(x, y, by.x = "id", by.y = "cd", all.x = T, all.y = T)
merged_energy <- merge(generation, imports, by = "datetime")
dim(merged_energy)

# =========================================================
# dplyr: select,filter, mutate, summarize 
# =========================================================
library(tidyverse)
### select
tmp <- select(merged_energy, biogas, biomass, geothermal, solar)
### one_of(), contains(), starts_with(), ends_with(), matches()
select(merged_energy, contains("hydro"), starts_with("bio"))

### filter
tmp <- filter(merged_energy, imports > 7000)
tmp <- filter(merged_energy, imports > 7000, natural_gas < 7000)

### mutat: create new variable
tmp <- mutate(long_merged_energy, log_usage = log(usage))
tmp <- mutate(long_merged_energy, log_usage = log(usage), usage2 = usage^2, usage3 = usage^3)
head(tmp)

### summarize
summarize(long_merged_energy, mean_cons = mean(usage, na.rm = T))
summarize(long_merged_energy, total = sum(usage, na.rm = T))

### pipe operator
long_merged_energy %>% 
  filter(source == "geothermal") %>% 
  select(-datetime) %>% 
  mutate(log_usage = log(usage)) %>% 
  summarize(mean_log_usage = mean(log_usage, na.rm = T))

# To refer to the manipulated dataframe, use .
merged_energy %>% 
  select(-datetime) %>% 
  mutate(total_usage = rowSums(., na.rm = T)) %>% 
  summarize(total_usage = sum(total_usage, na.rm = T))

### group by
long_merged_energy %>% 
  group_by(source) %>% 
  summarize(sum_usage = sum(usage, na.rm = T))

### _join
### left_join - keeps all observations in the first dataframe
merge(all.x = T)
### right_join - keeps all observations in the second dataframe
merge(all.y = T)
### full_join - keeps all observations in both dataframes
merge(all = T)
### inner_join - keeps only those observations that are matched in both datasets
merge(all = F)

tmp <- merge(generation, imports, by = "datetime", all = F)
tmp <- inner_join(generation, imports, by = "datetime")

# =========================================================
# data.table
# dt[i,j,by]
# i: On which rows; j: What to do; by: Grouped by what?
# =========================================================
library(data.table)
data_file <- here::here("R","data", "ca_energy_generation.csv")
generation_df <- read.csv(data_file, stringsAsFactors = F)
generation_dt <- fread(data_file)

### On which rows
generation_dt[wind > 4400]
generation_dt[wind > 4400 & mday(datetime) == 7]
generation_dt[natural_gas <= 5000 & large_hydro > 2000]
generation_dt[coal > 10 & solar > median(solar)]

### What to do
generation_dt[,wind + solar]
generation_dt[,3*wind + solar*biogas/2]
generation_dt[,newcol := 3*wind + solar*biogas/2]   # Add new feature. Add to the original table
generation_dt[,.(newcol = 3*wind + solar*biogas/2)]   # Add new feature. Not add to the original table
generation_dt[,.(mean(nuclear), mean(biogas))]
generation_dt[solar == 0, .(datetime, total_thermal = natural_gas + coal)]

### group by what
generation_dt[,mean(nuclear), by = mday(datetime)]
generation_dt[,.(mean_nuc = mean(nuclear), mean_wind = mean(wind)), 
              by = mday(datetime)]
generation_dt[hour(datetime) > 19,
              .(mean_nuc = mean(nuclear), mean_wind = mean(wind)), 
              by = mday(datetime)]

### reshape
long_ca_energy <- long_ca_energy %>%
  mutate(day = as_date(datetime),
         log_output = log(output)) %>%
  group_by(day) %>%
  mutate(total_daily_output = sum(output, na.rm = T)) %>% 
  ungroup() %>% 
  mutate(per_output = output/total_daily_output)   # using dplr

all_generation_long[,`:=`(day2 = as_date(datetime), 
                          log_output2 = log(value), 
                          per_output2 = value/sum(value)), 
                    by = day]   # using data.table

### set function
setnames(dt, "old", "new")   # set column names
setorder(dt, col1, -col2, ...)   # set row order
set(dt, i, j)   # set anything
dt[,col1 := 2*col2]   # set column

### special variable
# for .N: convenient
all_generation_long[,.N] 
all_generation_long[,.N, by = type]
# for .I: more advanced syntax
all_generation_long[,.I]

### set keys
key(generation_dt)   # check the current key
setkey(generation_dt, datetime)   # set key

### joins
imports_dt <- fread(here::here("data", "ca_energy_imports.csv"))
imports_dt
imports_dt[generation_dt, on = "datetime"]
imports_dt[generation_dt, on = "datetime", imports_gas := imports + i.natural_gas]   # i. stand for the first parameter


