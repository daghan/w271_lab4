setwd("~/Desktop/w271/Lab4")
rm(list=ls())
list.files()

# monthly series covers from 1990 to 2015
# use 1990 to 12/2014 data for training series
# use months of 2015 as test
# conduct a monthly, 11-month ahead forecast of the series in 2015

# ------------------------------------------------------------------------------------
# STEP 1: read the file and split into training and test set
df <- read.csv("Lab4-series2.csv",header = TRUE)
df.ts <- ts(df[,2], start = 1990, frequency = 12)
str(df.ts)
time(df.ts)

train.ts <- window(df.ts, start = c(1990,1), end=c(2014,12), freq = 12)
test.ts <- window(df.ts, start=c(2015,1), freq=12)

# ------------------------------------------------------------------------------------
# STEP 2: EDA and transformations

# 2.0 Plot the time series
library(fpp)
tsdisplay(train.ts)

# plot(train.ts)
# acf(train.ts, lag.max = 40)
# pacf(train.ts, lag.max = 40)

# The data is clearly non-stationary with some seasonality
# No sign of increasing variance, no need for log transformation

# 2.1 Let's also try STL decomposition to see the components in this series
fit.stl <- stl(df.ts, t.window=15, s.window="periodic", robust=TRUE)
plot(fit.stl)

# This series has a seasonal componet with a period of 12 months

# Q: if I take the 'trend' and do the first differencing, is the resulting series stationary?
# Q: similarly, if I take 'seasonal' and do seasonal differencing, what will happen to the resulting series?
# Q: if I take the original series, do first difference then do stl decomposition, what will happen?
# Q: if I take the original series, do seasonal difference then do stl decomposition, what will happen?
# Q: if I take the original series, do first and seasonal difference, then stl decomposition, what will happen?

# 2.2 First difference
tsdisplay(diff(train.ts))

# 2.3 Seasonal difference
tsdisplay(diff(train.ts, 12))

# 2.4 Apply Both Seasonal and First difference
tsdisplay(diff(diff(train.ts), 12))

# Significant spikes in ACF at lag 12
# Significant spikes in PACF at lag 12, 24, 36 ...
# Suggesting a seasonal MA(1) component

# Significant spikes in ACF at lag 2, (3,4), 5
# Significant spikes in PACF at lag 2, 5
# Suggesting a nonseasonal MA(2) component


# ------------------------------------------------------------------------------------
# STEP 3: Manual Model Fit

# Start with ARIMA(0,1,2)(0,1,1)[12] and try different variations
fit.test <- Arima(train.ts, order=c(1,1,0), seasonal=c(0,1,1)); fit.test; tsdisplay(residuals(fit.test))

# SUMMARY
# ARIMA                 AIC       AICc     BIC      ACF spikes   PACF spikes
# (0,1,1)(0,1,1)[12]   -97.78    -97.69   -86.8     2,3,5,7...   2,3,5
#*(0,1,2)(0,1,1)[12]  -108.77   -108.62   -94.13    3,5          3,5
# (0,1,3)(0,1,1)[12]  -109.62   -109.41   -91.33    5,20,36      5
# (0,1,4)(0,1,1)[12]  -109.61   -109.31   -87.66    5,20,36      5
#*(0,1,5)(0,1,1)[12]  -120.65   -120.25   -95.03    36           NONE
# (0,1,6)(0,1,1)[12]  -120.24   -119.72   -90.96    36           NONE

# (0,1,1)(0,1,2)[12]  -101.51   -101.37   -86.88    2,3,5        2,3,5,20
# (0,1,2)(0,1,2)[12]  -111.75   -111.54   -93.45    3,5          3,5,22
# (0,1,2)(0,1,3)[12]  -111.63   -111.33   -89.67    3,5          3,5

# (1,1,1)(0,1,1)[12]  -122.71   -122.57  -108.08    5            5,20
# (1,1,2)(0,1,1)[12]  -125.81   -125.6   -107.52    20,36        20
# (1,1,3)(0,1,1)[12]  -123.83   -123.53  -101.87    20,26        20
# (1,1,1)(0,1,2)[12]  -122.23   -122.01  -103.93    5            5,20
#*(2,1,1)(0,1,1)[12]  -125.86   -125.64  -107.56    20,36        20
# (3,1,1)(0,1,1)[12]  -123.86   -123.56  -101.9     20,36        20
# (2,1,1)(1,1,1)[12]  -125.55   -125.25  -103.59    20           20
# (1,1,1)(1,1,1)[12]  -122.4    -122.9   -104.1     5            5,20
# (2,1,1)(0,1,2)[12]  -125.38   -125.08  -103.42    20           NONE
# (2,1,2)(0,1,1)[12]  -123.86   -123.56  -101.9     20,36        20

# (2,1,1)(2,1,0)[12]   -75.82    -75.52   -53.86    20,24,34,36  20,24,36
# (1,1,1)(1,1,0)[12]   -61.08    -60.94   -46.44    24,34        24,36 
# (1,1,2)(2,1,0)[12]   -75.96    -75.66   -54       20,24,34,36  20,24,36
# (1,1,1)(1,1,1)[12]  -122.4    -122.19  -104.1     5            5,20
# (1,1,0)(0,1,1)[12]   -99.18    -99.1    -88.21    2,3,5,36     2,3,5,20

# ------------------------------------------------------------------------------------
# STEP 4: Check with Auto-Selection
auto.arima(train.ts)
# ARIMA(1,1,2)(1,0,0)[12]  AIC=-12.38   AICc=-12.17   BIC=6.13
# Obviously not good due to the shortcuts

# Turn off the shortcuts and try again
auto.arima(train.ts, stepwise = FALSE, approximation = FALSE)
# ARIMA(2,1,1)(1,0,0)[12] AIC=-12.45   AICc=-12.25   BIC=6.05
# Stll not good

auto.arima(train.ts, lambda=0, d=1, D=1, max.order=5, stepwise=FALSE, approximation=FALSE)
# ARIMA(2,1,1)(2,1,0)[12] AIC=-1072.33   AICc=-1072.03   BIC=-1050.37
# NOTE: why the AIC, AICc, BIC different from the ones by Arima()????? 
# COMMENT: the different is due to "lambda=0"

# It seems the Auto-Selection does not work well in this case

# ------------------------------------------------------------------------------------
# STEP 5: Mean Abosolute Percentage Error (MAPE) on the test sample
# Candidate Models
# ARIMA                 AIC          AICc         BIC        ACF spikes  PACF spikes  RMSE       MAPE
# (0,1,5)(0,1,1)[12]  -120.65      -120.25       -95.03      36          NONE         0.378      5.758
# (0,1,6)(0,1,1)[12]  -120.24      -119.72       -90.96      36          NONE         0.343      5.488 
# (1,1,1)(0,1,1)[12]  -122.71      -122.57      -108.08***** 5           5,20         0.247***** 4.173*****
# (1,1,2)(0,1,1)[12]  -125.81****  -125.6 ****  -107.52***   20,36       20           0.266      4.47
# (1,1,3)(0,1,1)[12]  -123.83      -123.53      -101.87      20,26       20           0.256****  4.307**
# (1,1,1)(0,1,2)[12]  -122.23      -122.01      -103.93*     5           5,20         0.265*     4.256***             
# (2,1,1)(0,1,1)[12]  -125.86***** -125.64***** -107.56****  20,36       20           0.266      4.464
# (3,1,1)(0,1,1)[12]  -123.86*     -123.56*     -101.9       20,36       20           0.257***   4.324*
# (2,1,1)(1,1,1)[12]  -125.55***   -125.25***   -103.59      20          20           0.29       4.55
# (1,1,1)(1,1,1)[12]  -122.4       -122.9       -104.1 **    5           5,20         0.261**    4.237****
# (2,1,1)(0,1,2)[12]  -125.38**    -125.08**    -103.42      20          NONE         0.296      4.581
# (2,1,2)(0,1,1)[12]  -123.86      -123.56      -101.9       20,36       20           0.258      4.347

# METHOD 1: RMSE from HA
getRMSE <- function(x,h,...)
{
  train.end <- time(x)[length(x)-h]
  test.start <- time(x)[length(x)-h+1]
  train <- window(x,end=train.end)
  test <- window(x,start=test.start)
  fit <- Arima(train,...)
  fc <- forecast(fit,h=h)
  return(round(accuracy(fc,test)[2,"RMSE"],3))
}

# METHOD 2: MAPE (Mean Abosolute Percentage Error)
# rowMeans(abs((actual-predicted)/actual) * 100)
getMAPE <- function(x,h,...)
{
  train.end <- time(x)[length(x)-h]
  test.start <- time(x)[length(x)-h+1]
  train <- window(x,end=train.end)
  test <- window(x,start=test.start)
  fit <- Arima(train,...)
  fc <- forecast(fit,h=h)
  return(round(mean(abs((test.ts-fc$mean)/fc$mean)*100),3))
}

getRMSE(df.ts,h=11,order=c(2,1,2),seasonal=c(0,1,1),lambda=0)
getMAPE(df.ts,h=11,order=c(2,1,2),seasonal=c(0,1,1),lambda=0)

# ------------------------------------------------------------------------------------
# STEP 6: Test the Selected Model

# Choose the ARIMA(2,1,1)(0,1,1)[12] model with the lowest AIC, AICc 
fit <- Arima(train.ts, order=c(2,1,1), seasonal=c(0,1,1))
res <- residuals(fit)
tsdisplay(res)
Box.test(res, lag=16, fitdf=4, type="Ljung")  # p-value = 0.6722
Box.test(res, lag=36, fitdf=6, type="Ljung")  # p-value = 0.2596
# NOTE: What parameters to use ???????
# - We can ignore the 2 spikes outside the 95% significant limits, the residuals appear to be white noise.
# - A Ljung-Box test also shows that the residuals have no remaining auto-correlations.

# Choose the ARIMA(1,1,1)(0,1,1)[12] model with the lowest BIC and best out-of-sample performance
# Choose the ARIMA(2,1,1)(0,1,1)[12] model with the lowest AIC, AICc 
fit <- Arima(train.ts, order=c(1,1,1), seasonal=c(0,1,1))
res <- residuals(fit)
tsdisplay(res)
Box.test(res, lag=16, fitdf=4, type="Ljung")  # p-value = 0.2188
Box.test(res, lag=36, fitdf=6, type="Ljung")  # p-value = 0.1085
# - We can ignore the 2 spikes outside the 95% significant limits, the residuals appear to be white noise.
# - A Ljung-Box test also shows that the residuals have no remaining auto-correlations.

# ------------------------------------------------------------------------------------
# STEP 7: 11-month ahead forecast of the series in 2015
# Choose the ARIMA(2,1,1)(0,1,1)[12] model with the lowest AIC, AICc 
fit <- Arima(train.ts, order=c(2,1,1), seasonal=c(0,1,1));fit
plot(forecast(fit), ylab="H02 sales (million scripts)", xlab="Year")

# Choose the ARIMA(1,1,1)(0,1,1)[12] model with the lowest BIC and best out-of-sample performance
fit <- Arima(train.ts, order=c(1,1,1), seasonal=c(0,1,1));fit
plot(forecast(fit), ylab="H02 sales (million scripts)", xlab="Year")
