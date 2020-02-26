# =========================================================
# Loop
# =========================================================
### for
gapminder <- read.csv(here::here("data", "gapminder5.csv"))
head(gapminder)
gapminder$country <- as.character(gapminder$country)
gapminder$continent <- as.character(gapminder$continent)

### recover the GDP for each country
obs <- 1:nrow(gapminder)
for (i in obs) { # the function to repeat is enclosed in braces {}
  gapminder[i, "gdp"] <- gapminder[i, "pop"] * gapminder[i, "gdpPercap"]
}

### recover the GDP for each country
for (i in years) {
  mean_le <- mean(gapminder$lifeExp[gapminder$year == i], 
                  na.rm = T)
  print(paste0(i, ": ", mean_le))
}

### while
i <-  1987 # define the interator

while (i <= 2002) {
  sd_le <- sd(gapminder$lifeExp[gapminder$year == i])
  print(paste0(i, ": ", sd_le)
  )
  i <- i + 5 # increase the iterator by the interval between years
}

# =========================================================
# apply
# =========================================================
vars <- gapminder[, c("lifeExp", "pop", "gdpPercap")]
apply(vars, 2, mean)

### lapply returns a list, sapply returns a simplified list
lapply(gapminder, mean)
sapply(years, function(x) mean(gapminder$lifeExp[gapminder$year == x]))

# =========================================================
# if
# =========================================================
if (random_year > 1977) {
  print(random_year)
}

# =========================================================
# function
# =========================================================
my_function <- # give the function a name
  function(x, y) { # arguments for the function go inside the parentheses
    # the expressions do in the braces
  return(object)
  }

