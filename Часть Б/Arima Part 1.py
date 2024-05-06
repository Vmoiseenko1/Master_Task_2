import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as scs
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn import metrics
from math import sqrt
import openpyxl
import scipy
from openpyxl import load_workbook
import math
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from numpy.polynomial import Polynomial
from scipy.signal import savgol_filter
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.arima.model import ARIMA
import statistics

df = pd.read_excel('C:/Users/USA_GAS.xls', sheet_name=2, index_col='Year')
s = int(len(df)*0.8) # Длина обучающей выборки (80% всех наблюдений)
df_test = df[0:s]
# FORMAT = ['Year', 'Value']
# df_selected = df[FORMAT] # датафрэйм для добычи газа в США с 1971 по 1990
df_for_rolling = pd.read_excel('C:/Users/USA_GAS.xls', sheet_name=2, index_col='Year')
s = int(len(df)*0.8)
df_train = df[0:s]
df_test = df[s:]

def t_t2_create(df):
    t = list(range(1, len(df) + 1))
    def t_2(x):
        s = []
        for i in list(range(1, len(x) + 1)):
            s.append(i ** 2)
        return s


    t = pd.DataFrame(t)
    t_sq = pd.DataFrame(t_2(df))
    col_t = list(t.columns)
    col_t_sq = list(t_sq.columns)
    col_t[0] = 't'
    col_t_sq[0] = 't^2'
    t.columns = col_t
    t_sq.columns = col_t_sq
    t.set_index(df.index, inplace=True)
    t_sq.set_index(df.index, inplace=True)
    df = pd.concat([df, t, t_sq], axis=1)
    print(df)
    return df

df = t_t2_create(df)
#print(df)

df1diff = df.diff(periods=1).dropna()
df2diff = df.diff(periods=2).dropna()
df12diff = df.diff(periods=12).dropna()

all_values_list = list(df['Value'])

first_difference = []
for i in range(1, len(all_values_list)):
    first_difference.append(round((all_values_list[i]-all_values_list[i-1]), 2))

second_difference = []
for i in range(2, len(all_values_list)):
    second_difference.append(round((all_values_list[i]-all_values_list[i-2]), 2))

# Для ряда сезонных разностей (берем 12)
twelve_difference = []
for i in range(12, len(all_values_list)):
    twelve_difference.append(round((all_values_list[i]-all_values_list[i-12]), 2))

# Для ряда сезонных разностей (берем 13!!!!! первая разность)
twenty_fourth_difference = []
for i in range(13, len(all_values_list)):
    twenty_fourth_difference.append(round((all_values_list[i]-all_values_list[i-13]), 2))

def diki_fuller(df):
    test = adfuller(df)
    print('adf: ', test[0])
    print('p-value: ', test[1])
    print('Critical values: ', test[4])
    if test[0] > test[4] ['5%']:
        print('Есть единичные корни, ряд не стационарен')
    else:
        print('Единичных корней нет, ряд стационарен')
diki_fuller(twenty_fourth_difference)

def kpss_test(series, **kw):
    statistic, p_value, n_lags, critical_values = kpss(series, **kw)
    # Format output
    print(f'KPSS Statistic: {statistic}')
    print(f'p-value: {p_value}')
    print(f'num lags: {n_lags}')
    print('Critical Values:')
    for key, value in critical_values.items():
        print(f'   {key} : {value}')
    print(f'Result: The series is {"not " if p_value < 0.05 else ""}stationary')
kpss_test(twenty_fourth_difference)

def tsplot(y, lags=None, figsize=(15,7), style='bmh'):
    #""ACF и PACF""
    if not isinstance(y, pd.Series): #Ставим формат Series для ряда
        y = pd.Series(y)

    with plt.style.context(style):

        fig = plt.figure(figsize=figsize)

        layout = (2, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))

        y.plot(ax = ts_ax, color = 'black')
        ts_ax.set_title('Time Series Analysis Plots')

        sm.graphics.tsa.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5, color = 'red')
        sm.graphics.tsa.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5, method='ywm', color = 'red')

        plt.tight_layout()
        plt.show()
    return

np.random.seed(1)
tsplot(twenty_fourth_difference, lags=30)

## Перебор параметров и оценка качества прогноза ARIMA
def prognoz_quality(y_pred, y_true):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    n = len(y_pred)
    mape = 1 / n * sum(abs(y_pred - y_true) / y_true)
    rmse = np.sqrt(1 / n * sum((y_pred - y_true) ** 2))
    accuracy = round(1 - rmse/np.average(y_true),2)
    print("mape: ", mape)
    print("rmse: ", rmse)
    print("accuracy: ", accuracy)

def MAPE(y_pred, y_true): # с ошибкой
    n = len(y_pred)
    return 1 / n * sum(abs(y_pred - y_true) / y_true)

def RMSE(y_pred, y_true): ## среднеквадратическое отклонение
    n = len(y_pred)
    return np.sqrt(1 / n * sum((y_pred - y_true) ** 2))

# def rolling_csv_sarima(y, folds, horizon):
#     mape_list = []
#     for combination in range(0, len(parameters_list)):
#         try:
#             mape_mean_list = []
#             for i in range(folds):
#                 train_ind_start, train_ind_end = 0, len(y)-(i+1)*horizon
#
#                 test_ind_start, test_ind_end = train_ind_end, train_ind_end+horizon
#
#                 y_train = y[train_ind_start:train_ind_end]
#                 y_test = y[test_ind_start:test_ind_end]
#
#                 model_for_rolling = sm.tsa.statespace.SARIMAX(y_train, order=(parameters_list[combination][0], d, parameters_list[combination][1]), seasonal_order=(parameters_list[combination][2], D, parameters_list[combination][3], 12)).fit(disp=-1)
#                 y_backed = model_for_rolling.forecast(horizon)
#                 mape_i = MAPE(y_true=y_test, y_pred= y_backed)
#                 mape_mean_list.append(mape_i)
#
#             mape_mean = statistics.mean(mape_mean_list)
#             mape_list.append(mape_mean)
#             print(f'MAPE on combination {parameters_list[combination]}:{round(mape_mean, 4)}')
#         except UserWarning:
#             print('wrong parametrs:', parameters_list[i])
#             continue
#     print("min ьуфт_ьфзу", min(mape_list))mape_list



# def predict_arima(df, pdqPDQ, col="Value", train=0.8):
#     """Прогноз для SARIMA(p, d, q, P, D, Q) для ARIMA P, D, Q равны 0."""
#     x = df["t"].values
#     y = df[col]
#
#     info = []
#
#     for p, d, q, P, D, Q in pdqPDQ:
#         model_info = {}
#         len_train = int(len(y) * train)
#
#         y_train = y[:len_train]
#         y_test = y[len_train:]
#         x_test = x[len_train-1:]
#
#         model = ARIMA(y_train, order=(p, d, q), seasonal_order=(P, D, Q, 12)).fit()
#         y_predicted = model.predict(start=x_test[0], end=x_test[-2])
#
#         model_info["pdq"] = (p, d, q)
#         model_info["rmse"] = round(RMSE(y_predicted.to_numpy(), y_test.values), 3)
#         model_info["mape"] = round(MAPE(y_predicted.to_numpy(), y_test.values), 3)
#
#         print()
#         print("--- ARIMA (" + str(p) + ", " + str(d) + ", " + str(q) + ") ---")
#         print("RMSE: " + str(model_info["rmse"]))
#         print("MAPE: " + str(model_info["mape"]))
#         print()
#
#         info.append(model_info)
#
#         with plt.style.context("bmh"):
#             plt.figure(figsize=(8, 4))
#             plt.plot(y_test, linewidth=6, color="#64adf1", label="Исходный ряд")
#             plt.plot(y_predicted, linewidth=3, color="#f66a98", label="Прогноз")
#             plt.xlabel("Year")
#             plt.ylabel("Value")
#             plt.legend(loc='upper center', fontsize=12, ncol=2)
#             plt.title("SARIMA (" + str(p) + ", " + str(d) + ", " + str(q) + ", " + str(P) + ", " + str(D) + ", " + str(Q) + ")")
#             plt.show()
#
#     return info
# predict_arima(df, pdqPDQ, col="Value", train=0.8)


# ARIMA  критериями AIC BIC
p = 1
d = 3
q = 2
P = 0
D = 2
Q = 1
parameters_list = []
for pp in range(0, p+1):
    #for qq in range(2, 5):
    for PP in range(0, P+1):
        for QQ in range(0, Q+1):
            parameters_list.append([pp, 5, PP, QQ])


print(parameters_list)
results=[]
for i in range(0, len(parameters_list)):
    try:
        model = sm.tsa.SARIMAX(all_values_list, order=(parameters_list[i][0], parameters_list[i][1], parameters_list[i][2])).fit()

    except ValueError:
        print('wrong parametrs:', parameters_list[i])
        continue
    aic = model.aic
    results.append([parameters_list[i], model.aic])
print(results)



# АРИМА
y_test = all_values_list[0:int(len(all_values_list)*0.8)]
model_for_prognoz = sm.tsa.ARIMA(y_test, order=(1,2,1)).fit()
y_prognoz = list(model_for_prognoz.forecast(len(all_values_list)-int(len(all_values_list)*0.8)))
print(type(y_prognoz))
print(type(y_test))
y_new = y_test+ y_prognoz
print(len(y_new), y_new)
with plt.style.context("bmh"):
  plt.figure(figsize=(8, 4))

  plt.plot(y_new, linewidth=2, color="#64adf1", label="")
  plt.plot(y_prognoz, linewidth=3, color="#f66a98", label="Прогноз")
  plt.xlabel("t")
  plt.ylabel("Value")
  plt.legend(loc='upper center', fontsize=12, ncol=2)
  plt.title("ARIMA (" + str(p) + ", " + str(d) + ", " + str(q)+ ")")
  plt.show()

prognoz_quality(y_prognoz, all_values_list[int(len(all_values_list)*0.8):])




#SARIMA
y_test = all_values_list[0:int(len(all_values_list)*0.8)]
model_for_prognoz = sm.tsa.statespace.SARIMAX(y_test, order=(p, d, q), seasonal_order=(P, D, Q, 12)).fit(disp=-1)
y_prognoz = list(model_for_prognoz.forecast(len(all_values_list)-int(len(all_values_list)*0.8)))
print(type(y_prognoz))
print(type(y_test))
y_new = y_test + y_prognoz
print(len(y_new), y_new)
with plt.style.context("bmh"):
    plt.figure(figsize=(8, 4))
    plt.plot(y_new, linewidth=2, color="#64adf1", label="")
    plt.plot(all_values_list, linewidth=2, color="green", label="")
    #plt.plot(y_prognoz, linewidth=3, color="#f66a98", label="Прогноз")
    plt.xlabel("t")
    plt.ylabel("Value")
    plt.legend(loc='upper center', fontsize=12, ncol=2)
    plt.show()

prognoz_quality(y_prognoz, all_values_list[int(len(all_values_list)*0.8):])

#rolling_csv_sarima(list(df_for_rolling['Value']), 3, 10)









