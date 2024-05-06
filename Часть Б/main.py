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

#from stldecompose import decompose, forecast
#from stldecompose.forecast_funcs import (naive, drift,mean, seasonal_naive)

df = pd.read_excel('C:/Users/USA_GAS.xls')
FORMAT = ['Year', 'Value']
df_selected = df[FORMAT] # датафрэйм для добычи газа в США с 1971 по 1990


def plot_df(df_selected, x, y, title="", style='bmh'):
    with plt.style.context(style):
        df_selected.plot(x, y, figsize=(16, 5))
        plt.gca().set(title=title)
        plt.show()
#plot_df(df_selected, "Year", "Value", title='Monthly gas production (billion cubic feet) in USA from 1971 to 1990')

my_columns = list(df_selected.columns)
my_columns[0] = 'Gas'
df_selected.columns = my_columns

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

## Выделим тренд, сезонность и остатки с помощью STL
# Мультипликативное разложение
result_mul = seasonal_decompose(df['Value'], model='multiplicative', extrapolate_trend='freq', period=12)

# Аддитивное разложение
result_add = seasonal_decompose(df['Value'], model='additive', extrapolate_trend='freq', period=12)

#Plot
#plt.rcParams.update({'figure.figsize': (10, 10)})
#result_mul.plot().suptitle('Multiplicative Decompose', fontsize = 22)
#result_add.plot().suptitle('Additive Decompose', fontsize = 22)
#plt.show()

df_reconstructed = pd.concat([result_mul.seasonal, result_mul.trend, result_mul.resid, result_mul.observed], axis=1)
df_reconstructed.columns = ['seasonal', 'trend', 'residuals', 'actual_values']
df_reconstructed.head()
#print(df_reconstructed)

## Периодограмма и разложение в ряд Фурье
fluctuations = df_reconstructed['actual_values'] - df_reconstructed['trend']

def period(df):
    a, b = [], []
    a.append(df.mean()) # Alpha_0
    b.append(0)

    for j in range(1, len(df)//2):
        p = 0
        q = 0
        for t in range(1, len(df) + 1):
            p = p + df[t-1] * math.cos(2 * math.pi * j * t / len(df))
            q = q + df[t-1] * math.sin(2 * math.pi * j * t / len(df))

        a.append(2 / len(df) * p) # коэффы Alpha_j (всего их T/2)
        b.append(2 / len(df) * q) # коэффы Beta_j (всего их T/2)

    T_2 = 0
    for t in range(1, len(df) + 1):
        T_2 += 1/len(df) * (-1)**t * df[t-1]

    a.append(T_2)
    b.append(0)

    periodogramma = []

    for i in range(len(a)):
        I_j = (a[i] ** 2 + b[i] ** 2) * len(df) // 2 #интенсивность для j-й гармоники
        periodogramma.append(I_j)

    fig, ax = plt.subplots()
    plt.plot(range(len(periodogramma)), periodogramma, c='Black')
    ax.legend(loc = 'upper center', fontsize = 25, ncol = 2)
    fig.set_figwidth(20)
    fig.set_figheight(6)
    fig.suptitle('Периодограмма')
    #plt.show()

    # Разложение исходного ряда в ряд Фурье
    Furie = []
    for t in range(1, len(df) + 1):
        x = 0
        for j in range(len(a)):
            x = x + a[j] * math.cos(2 * math.pi * j * t/len(df)) + b[j] * math.sin(2 * math.pi * j * t/len(df))
        Furie.append(x)
    return periodogramma
#return a, b, periodogramma, Furie
#y1 = list(fluctuations)
#y2 = period(fluctuations)[3]
#fig, ax = plt.subplots()
#plt.plot(range(len(y1)), y1, c = 'hotpink', linewidth = 4, label = 'Data')
#plt.plot(range(len(y2)), y2, c = 'Black', linewidth = 3, label = 'Forecast')
#ax.legend(loc = 'upper center', fontsize = 25, ncol = 2)
#fig.set_figwidth(20)
#fig.set_figheight(6)
#fig.suptitle("Представление в виде конечного ряда Фурье")
#plt.show()
#print(period(fluctuations))

## Модель линейного трендаS
x2 = df[['t']]
y2 = df['Value']
estimator2 = LinearRegression()
estimator2.fit(x2, y2)
y_pred2 = estimator2.predict(x2)
#print(f'Slope : {format(round(estimator2.coef_[0],2))}') #a1
#print(f'Intercept : {format(round(estimator2.intercept_,2))}') #const
#print(f'R^2 : {round(estimator2.score(x2,y2),2)}')

model_s = smf.ols('Value ~ t', data=df)
res1 = model_s.fit()
#print(res1.summary())

## Квадратичный тренд
y1 = df['Value']
x1 = df[['t', 't^2']]
estimator1 = LinearRegression()
estimator1.fit(x1, y1)
y_pred1 = estimator1.predict(x1)
#print(f'Slope 1 : {format(round(estimator1.coef_[0],2))}') #угол наклона перед t, a1
#print(f'Slope 2 : {format(round(estimator1.coef_[1],4))}') #угол наклона перед t^2, a2
#print(f'Intercept : {format(round(estimator1.intercept_,2))}') #константа
#print(f'R^2 : {round(estimator1.score(x1,y1),2)}')

model_sq = smf.ols('Value ~ t + t^2', data=df)
res2 = model_sq.fit()
#print(res2.summary())

# Выделим сезонные переменные (все месяца, кроме апреля)
def create_dummy(df):
    ## January
    y = []
    for i in list(range(len(df.index))):
        if i % 12 == 0:
            y.append(1)
        else:
            y.append(0)
    y = pd.DataFrame(y)
    y.set_index(df.index, inplace = True)
    January = y

    col_January = list(January.columns)

    col_January[0] = 'January'

    January.columns = col_January

    ## February
    f = []
    for i in list(range(len(df.index))):
        if i % 12 == 1:
            f.append(1)
        else:
            f.append(0)
    f = pd.DataFrame(f)
    f.set_index(df.index, inplace = True)
    February = f

    col_February = list(February.columns)

    col_February[0] = 'February'

    February.columns = col_February

    ## March
    m = []
    for i in list(range(len(df.index))):
        if i % 12 == 2:
            m.append(1)
        else:
            m.append(0)
    m = pd.DataFrame(m)
    m.set_index(df.index, inplace = True)
    March = m

    col_March = list(March.columns)

    col_March[0] = 'March'

    March.columns = col_March

    ## May
    ma = []
    for i in list(range(len(df.index))):
        if i % 12 == 4:
            ma.append(1)
        else:
            ma.append(0)
    ma = pd.DataFrame(ma)
    ma.set_index(df.index, inplace = True)
    May = ma

    col_May = list(May.columns)

    col_May[0] = 'May'

    May.columns = col_May

    ## June
    jn = []
    for i in list(range(len(df.index))):
        if i % 12 == 5:
            jn.append(1)
        else:
            jn.append(0)
    jn = pd.DataFrame(jn)
    jn.set_index(df.index, inplace = True)
    June = jn

    col_June = list(June.columns)

    col_June[0] = 'June'

    June.columns = col_June

    ## July
    jl = []
    for i in list(range(len(df.index))):
        if i % 12 == 6:
            jl.append(1)
        else:
            jl.append(0)
    jl = pd.DataFrame(jl)
    jl.set_index(df.index, inplace = True)
    July = jl

    col_July = list(July.columns)

    col_July[0] = 'July'

    July.columns = col_July

    ## August
    au = []
    for i in list(range(len(df.index))):
        if i % 12 == 7:
            au.append(1)
        else:
            au.append(0)
    au = pd.DataFrame(au)
    au.set_index(df.index, inplace = True)
    August= au

    col_August = list(August.columns)

    col_August[0] = 'August'

    August.columns = col_August


    ## September
    x = []
    for i in list(range(len(df.index))):
        if i % 12 == 8:
            x.append(1)
        else:
            x.append(0)
    x = pd.DataFrame(x)
    x.set_index(df.index, inplace = True)
    September = x

    col_September = list(September.columns)

    col_September[0] = 'September'

    September.columns = col_September

    ## October
    o = []
    for i in list(range(len(df.index))):
        if i % 12 == 9:
            o.append(1)
        else:
            o.append(0)
    o = pd.DataFrame(o)
    o.set_index(df.index, inplace = True)
    October = o

    col_October = list(October.columns)

    col_October[0] = 'October'

    October.columns = col_October

    ## November
    n = []
    for i in list(range(len(df.index))):
        if i % 12 == 10:
            n.append(1)
        else:
            n.append(0)
    n = pd.DataFrame(n)
    n.set_index(df.index, inplace = True)
    November = n

    col_November = list(November.columns)

    col_November[0] = 'November'

    November.columns = col_November

    ## December
    d = []
    for i in list(range(len(df.index))):
        if i % 12 == 11:
            d.append(1)
        else:
            d.append(0)
    d = pd.DataFrame(d)
    d.set_index(df.index, inplace = True)
    December = d

    col_December = list(December.columns)

    col_December[0] = 'December'

    December.columns = col_December

    df = pd.concat([df, January, February, March, May, June, July, August, September, October, November, December], axis = 1)
    print(df)
    return df

df = create_dummy(df)

## Линейный тренд с сезонными фиктивными переменными
x4 = df[['t', 'January', 'February', 'March','May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']]
y4 = df['Value']
estimator4 = LinearRegression()
estimator4.fit(x4, y4)
y_pred4 = estimator4.predict(x4)
#print(f'Slope 1 : {format(round(estimator4.coef_[0],2))}') #a1
#print(f'Slope 2 : {format(round(estimator4.coef_[1],4))}') #a2 January
#print(f'Slope 3 : {format(round(estimator4.coef_[2],4))}') #a3 February
#print(f'Slope 4 : {format(round(estimator4.coef_[3],4))}') #a4 March
#print(f'Slope 5 : {format(round(estimator4.coef_[4],4))}') #a5 May
#print(f'Slope 6 : {format(round(estimator4.coef_[5],4))}') #a6 June
#print(f'Slope 7 : {format(round(estimator4.coef_[6],4))}') #a7 July
#print(f'Slope 8 : {format(round(estimator4.coef_[7],4))}') #a8 August
#print(f'Slope 9 : {format(round(estimator4.coef_[8],4))}') #a9 September
#print(f'Slope 10 : {format(round(estimator4.coef_[9],4))}') #a10 October
#print(f'Slope 11 : {format(round(estimator4.coef_[10],4))}') #a11 November
#print(f'Slope 12 : {format(round(estimator4.coef_[11],4))}') #a12 December
#print(f'Intercept: {format(round(estimator4.intercept_,2))}') #const
#print(f'R^2 : {round(estimator4.score(x4, y4),2)}')

## Квадратичный тренд с сезонными фиктивными переменными
x3 = df[['t', 't^2', 'January', 'February', 'March','May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']]
y3 = df['Value']
estimator3 = LinearRegression()
estimator3.fit(x3, y3)
y_pred3 = estimator3.predict(x3)
#print(f'Slope 1: {format(round(estimator3.coef_[0],2))}') #a1
#print(f'Slope 2: {format(round(estimator3.coef_[1],4))}') #a2
#print(f'Slope 3: {format(round(estimator3.coef_[2],4))}') # угол наклона перед January (a3)
#print(f'Slope 4: {format(round(estimator3.coef_[3],4))}') # угол наклона перед February (a4)
#print(f'Slope 5: {format(round(estimator3.coef_[4],4))}') # угол наклона перед March (a5)
#print(f'Slope 6: {format(round(estimator3.coef_[5],4))}') # угол наклона перед May (a6)
#print(f'Slope 7: {format(round(estimator3.coef_[6],4))}') # угол наклона перед June (a7)
#print(f'Slope 8: {format(round(estimator3.coef_[7],4))}') # угол наклона перед July (a8)
#print(f'Slope 9: {format(round(estimator3.coef_[8],4))}') # угол наклона перед August (a9)
#print(f'Slope 10: {format(round(estimator3.coef_[9],4))}') # угол наклона перед September (a10)
#print(f'Slope 11: {format(round(estimator3.coef_[10],4))}') # угол наклона перед October (a11)
#print(f'Slope 12: {format(round(estimator3.coef_[11],4))}') # угол наклона перед November (a12)
#print(f'Slope 13: {format(round(estimator3.coef_[12],4))}') # угол наклона перед December (a13)
#print(f'Intercept : {format(round(estimator3.intercept_,2))}') # константа
#print(f'R^2 : {round(estimator3.score(x3, y3),2)}')

## Графики с трендами
# Линейный тренд
#fig, axs = plt.subplots(2, 2, figsize=(28,20))
#axs[0,0].plot(y_pred2, color='red')
#axs[0,0].plot(list(y2), color='blue')
#axs[0,0].set_title('Linear trend')
#axs[0,0].set_xlabel('Year')
#axs[0,0].set_ylabel('Monthly Gas Production in USA')

#axs[0,1].plot(y_pred1, color='red')
#axs[0,1].plot(list(y1), color='blue')
#axs[0,1].set_title('Quadratic trend')
#axs[0,1].set_xlabel('Year')
#axs[0,1].set_ylabel('Monthly Gas Production in USA')

#axs[1,0].plot(y_pred4, color='red')
#axs[1,0].plot(list(y4), color='blue')
#axs[1,0].set_title('Linear trend with dummy variables')
#axs[1,0].set_xlabel('Year')
#axs[1,0].set_ylabel('Monthly Gas Production in USA')

#axs[1,1].plot(y_pred3, color='red')
#axs[1,1].plot(list(y3), color='blue')Ы
#axs[1,1].set_title('Quadratic trend with dummy variables')
#axs[1,1].set_xlabel('Year')
#axs[1,1].set_ylabel('Monthly Gas Production in USA')

#plt.show()

# Проверим стат значимость моделей
# Для квадратичного тренда с фикт пер
model_sqf = smf.ols('Value ~ t + t^2 + January + February + March + May + June + July + August + September + October + November + December', data=df)
res3 = model_sqf.fit()
#print(res3.summary())

# Для линейного тренда с фикт пер
model_lf = smf.ols('Value ~ t + January + February + March + May + June + July + August + September + October + November + December', data=df)
res4 = model_lf.fit()
#print(res4.summary())

## Прогнозирование
s = int(len(df)*0.8) # Длина обучающей выборки (80% всех наблюдений)

# Разделим выборки на train/test
Xtrn1 = x1[0:s]
Xtrn2 = x2[0:s]
Xtrn3 = x3[0:s]
Xtrn4 = x4[0:s]

Xtest1 = x1[s:]
Xtest2 = x2[s:]
Xtest3 = x3[s:]
Xtest4 = x4[s:]
y = df[["Value"]]
Ytest = y[s:]
Ytrn = y[0:s]

############### Точечный прогноз

## Оценка качества модели
## Метрики качества прогноза
def MAPE(y_pred, y_true): # с ошибкой
    n = len(y_pred)
    return 1 / n * sum(abs(y_pred - y_true) / y_true)

def RMSE(y_pred, y_true): ## среднеквадратическое отклонение
    n = len(y_pred)
    return np.sqrt(1 / n * sum((y_pred - y_true) ** 2))

## Создаем временные структуры
TestModels = pd.DataFrame()
tmp = {} # словарь для параметров после обучения
score = [] # список для результатов

x1 = df[['t']] # linear
x2 = df[['t', 't^2']] # quadratic
x3 = df[['t', 'January', 'February', 'March','May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']] # linear + dummy
x4 = df[['t', 't^2', 'January', 'February', 'March','May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']] # quadr + dummy
y = df[['Value']] # целевая переменная

s = int(len(df)*0.8) # Длина обучающей выборки (80% всех наблюдений)

# Разделим выборки на train/test
Xtrn1 = x1[0:s]
Xtrn2 = x2[0:s]
Xtrn3 = x3[0:s]
Xtrn4 = x4[0:s]

Xtest1 = x1[s:]
Xtest2 = x2[s:]
Xtest3 = x3[s:]
Xtest4 = x4[s:]

Ytest = y[s:]
Ytrn = y[0:s]

model = LinearRegression(fit_intercept=True)
trends = ['Linear', 'Quadratic', 'Linear with Dummy', 'Quadratic with Dummy']

for trend in trends:
    if trend == trends[0]:
        tmp['Model'] = trends[0]

        model.fit(Xtrn1, Ytrn)
        y_pred1 = model.predict(Xtest1)
        tmp['RMSE'] = round(RMSE(y_pred1, Ytest.values)[0],2)
        tmp['MAPE'] = round(MAPE(y_pred1, Ytest.values)[0],2)
        tmp['accuracy'] = round(1 - tmp['RMSE']/np.average(Ytest), 2)
        tmp['coefficient'] = round(model.coef_[0][0],4)

    if trend == trends[1]:
        tmp['Model'] = trends[1]
        model.fit(Xtrn2, Ytrn)
        y_pred2 = model.predict(Xtest2)
        tmp['RMSE'] = round(RMSE(y_pred2, Ytest.values)[0],2)
        tmp['MAPE'] = round(MAPE(y_pred2, Ytest.values)[0],2)
        tmp['accuracy'] = round(1 - tmp['RMSE']/np.average(Ytest),2)

        for i in list(range(len(model.coef_[0]))):
            model.coef_[0][i] = round(model.coef_[0][i],4)

        tmp['coefficient'] = model.coef_[0]


    if trend == trends[2]:
       tmp['Model'] = trends[2]
       model.fit(Xtrn3, Ytrn)
       y_pred3 = model.predict(Xtest3)

       tmp['RMSE'] = round(RMSE(y_pred3, Ytest.values)[0],2)
       tmp['MAPE'] = round(MAPE(y_pred3, Ytest.values)[0],2)
       tmp['accuracy'] = round(1 - tmp['RMSE']/np.average(Ytest),2)

       for i in list(range(len(model.coef_[0]))):
           model.coef_[0][i] = round(model.coef_[0][i],4)

       tmp['coefficient'] = model.coef_[0]
       #print('Coefficients for Linear with Dummy: ' + str(tmp['coefficient']))

    if trend == trends[3]:
       tmp['Model'] = trends[3]
       model.fit(Xtrn4, Ytrn)
       y_pred4 = model.predict(Xtest4)

       tmp['RMSE'] = round(RMSE(y_pred4, Ytest.values)[0],2)
       tmp['MAPE'] = round(MAPE(y_pred4, Ytest.values)[0],2)
       tmp['accuracy'] = round(1 - tmp['RMSE']/np.average(Ytest),2)

       for i in list(range(len(model.coef_[0]))):
           model.coef_[0][i] = round(model.coef_[0][i],4)

       tmp['coefficient'] = model.coef_[0]

       print('Coefficients for Quadratic with Dummy: ' + str(tmp['coefficient']))
       print(f'Intercept : {format(round(estimator4.intercept_, 2))}')
       df_prognoz = pd.read_excel("C:/Users/USA_GAS.xls", sheet_name=1)
       FORMAT = ['Year', 'Value']
       df_prognoz = df_prognoz[FORMAT]
       df_prognoz = t_t2_create(df_prognoz)
       df_prognoz = create_dummy(df_prognoz)
       y_prognoz = model.predict(df_prognoz[
                                     ["t", "t^2",'January', 'February', 'March', 'May', 'June', 'July', 'August', 'September',
                                      'October', 'November', 'December']][131:146])
       print(y_prognoz)
       for i in range(1, 16):
           df_prognoz.at[131 + i, 'value'] = y_prognoz[i - 1][0]
       print(df_prognoz)
       plot_df(df_prognoz, "Year", "value", title='Forecast')
    #TestModels = TestModels.append([tmp])
#TestModels.set_index('Model', inplace = True)
#print(TestModels)

# Экспоненциальный тренд
y = list(df['Value'])
newlis = []
for yznash in y:
    newlis.append(np.log(yznash))
print(y)
print(newlis)
df['logvalue'] = newlis
y = df[['logvalue']]

Ytest = y[s:]
Ytrn = y[0:s]
model = LinearRegression(fit_intercept=True)

for trend in trends:
    if trend == trends[0]:
        tmp['Model'] = trends[0]

        model.fit(Xtrn1, Ytrn)
        y_pred1 = model.predict(Xtest1)
        tmp['RMSE'] = round(RMSE(y_pred1, Ytest.values)[0],2)
        tmp['MAPE'] = round(MAPE(y_pred1, Ytest.values)[0],2)
        tmp['accuracy'] = round(1 - tmp['RMSE']/np.average(Ytest), 2)
        tmp['coefficient'] = round(model.coef_[0][0],4)

    if trend == trends[1]:
        tmp['Model'] = trends[1]
        model.fit(Xtrn2, Ytrn)
        y_pred2 = model.predict(Xtest2)
        tmp['RMSE'] = round(RMSE(y_pred2, Ytest.values)[0],2)
        tmp['MAPE'] = round(MAPE(y_pred2, Ytest.values)[0],2)
        tmp['accuracy'] = round(1 - tmp['RMSE']/np.average(Ytest),2)

        for i in list(range(len(model.coef_[0]))):
            model.coef_[0][i] = round(model.coef_[0][i],4)

        tmp['coefficient'] = model.coef_[0]


    if trend == trends[2]:
        tmp['Model'] = trends[2]
        model.fit(Xtrn3, Ytrn)
        y_pred3 = model.predict(Xtest3)

        tmp['RMSE'] = round(RMSE(y_pred3, Ytest.values)[0],2)
        tmp['MAPE'] = round(MAPE(y_pred3, Ytest.values)[0],2)
        tmp['accuracy'] = round(1 - tmp['RMSE']/np.average(Ytest),2)

        for i in list(range(len(model.coef_[0]))):
            model.coef_[0][i] = round(model.coef_[0][i],4)

        tmp['coefficient'] = model.coef_[0]
        print('Coefficients for Linear with Dummy: ' + str(tmp['coefficient']))

    if trend == trends[3]:
        tmp['Model'] = trends[3]
        model.fit(Xtrn4, Ytrn)
        y_pred4 = model.predict(Xtest4)

        tmp['RMSE'] = round(RMSE(y_pred4, Ytest.values)[0],2)
        tmp['MAPE'] = round(MAPE(y_pred4, Ytest.values)[0],2)
        tmp['accuracy'] = round(1 - tmp['RMSE']/np.average(Ytest),2)

        for i in list(range(len(model.coef_[0]))):
            model.coef_[0][i] = round(model.coef_[0][i],4)

        tmp['coefficient'] = model.coef_[0]

        print('Coefficients for Quadratic with Dummy: ' + str(tmp['coefficient']))
        print(f'Intercept : {format(round(estimator4.intercept_, 2))}')

    TestModels = TestModels.append([tmp])
print(TestModels['accuracy'])

# Спирмен
def spirmen():
    teta = []
    sortedvalues = sorted(df['Value'].to_list())
    for i in df['Value']:
        teta.append(sortedvalues.index(i)+1)
    tlist = df['t'].to_list()
    squared_dif = 0
    for value in range(len(tlist)):
        squared_dif += ((tlist[value]-teta[value])**2)
    etta = 1 - ((6*squared_dif)/(len(tlist)*((len(tlist))**2 - 1)))
    stat = etta / math.sqrt((1 - etta**2)/(len(tlist)-2))
    tstat = scipy.stats.t.ppf(0.975, len(tlist)-2)
    print('Коэффициент ранговой корреляции Спирмена: ' + str(etta))
    print('Статистика на его основе: ' + str(abs(stat)))
    print('t - статистика с ' + str(len(tlist) - 2) + ' степенями свободы: ' + str(tstat))
    if abs(stat)<tstat:
        print('Тренда нет:', abs(stat),'<', tstat)
    else:
        print('Тренд есть:', abs(stat),'>', tstat)

spirmen()

# Полномиальное сглаживание (Savitzky-Golay Filter)
smoothed_data = savgol_filter(np.ravel(df['Value']), 7, 1)
plt.plot(df['Year'], df['Value'], color='blue', linewidth=3)
plt.plot(df['Year'], smoothed_data, color='red', linewidth=2)
plt.show()

# Экспоненциальное сглаживание
df_exp = pd.read_excel("C:/Users/USA_GAS.xls", sheet_name = 2, index_col='Year')
df_exp1 = pd.read_excel("C:/Users/USA_GAS.xls", sheet_name = 2)
df_exp.index = pd.DatetimeIndex(df_exp.index)
exp_smooth = SimpleExpSmoothing(df_exp, initialization_method='estimated').fit(smoothing_level=0.8)
print('predict', list(exp_smooth.predict(0).values))
print(exp_smooth.summary())
df_exp1['SES'] = list(exp_smooth.fittedvalues)
plt.figure(figsize=(16,8))
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Alpha=0.8')
line_1, =plt.plot(df_exp1["Value"], color='blue')
line_2, =plt.plot(df_exp1['SES'], color='red')
plt.legend([line_1, line_2], ['Initial Series', 'Smooth Series'])
plt.show()

# trend_features = pd.DataFrame(index=df.index)
# trend_features['t'] = [x for x in range(1, len(df)+1)]
# trend_features['t2'] = [x**2 for x in range(1, len(df)+1)]
#
# df['month'] = [x.month_name() for x in df.reset_index()['Year']]
# print(df['month'])
# seasonal_features = pd.get_dummies(df['month'], drop_first = True) # переменные сезонности
#
# X = pd.concat([trend_features, seasonal_features], axis=1)
# y = df['Value']
# y_log = np.log(y)
#
# data = pd.concat([X, y], axis=1) #Dataframe для кросс-валидации
#
# # выделим тестовую и обучающую выбори
# horizon = 12
# X_train = X.head(-horizon)
# y_train = y.head(-horizon)
# y_train_log = y_log.head(-horizon)
#
# X_test = X.tail(horizon)
# y_test = y.tail(horizon)
# Y_test_log = y_log.tail(horizon)
#
# # Построение моделей
# # Списки факторов
# features_list1 = ['t']
# features_list2 = ['t', 't2']
# #features_list3 = features_list1 + seasonal_f

# Holt-Winters
### Сглаживание
df_for_last = pd.read_excel("C:/Users/USA_GAS.xls", sheet_name=2, index_col='Year')

df1 = df_for_last.copy()
add_add = ExponentialSmoothing(df1, seasonal_periods=12, trend='add', seasonal='add', use_boxcox=True, initialization_method='estimated',).fit()
df1['HW'] = list(add_add.fittedvalues)
plt.figure(figsize=(16,8))
plt.xlabel('Year')
plt.ylabel('Value')
plt.title('Additive Trend & Additive Seasonality')
line_1, = plt.plot(df1['Value'], color='blue')
line_2, = plt.plot(df1['HW'], color='red')
plt.legend([line_1, line_2], ['Initial Series', 'Smooth Series'])
plt.show()

df2 = df_for_last.copy()
add_mul = ExponentialSmoothing(df2, seasonal_periods=12, trend='add', seasonal='mul', use_boxcox=True, initialization_method='estimated',).fit()
df2['HW'] = list(add_mul.fittedvalues)
plt.figure(figsize=(16,8))
plt.xlabel('Year')
plt.ylabel('Value')
plt.title('Additive Trend & Multiplicative Seasonality')
line_1, = plt.plot(df2['Value'], color='blue')
line_2, = plt.plot(df2['HW'], color='red')
plt.legend([line_1, line_2], ['Initial Series', 'Smooth Series'])
plt.show()

df3 = df_for_last.copy()
mul_add = ExponentialSmoothing(df3, seasonal_periods=12, trend='mul', seasonal='add', use_boxcox=True, initialization_method='estimated',).fit()
df3['HW'] = list(mul_add.fittedvalues)
plt.figure(figsize=(16,8))
plt.xlabel('Year')
plt.ylabel('Value')
plt.title('Multiplicative Trend & Additive Seasonality')
line_1, = plt.plot(df3['Value'], color='blue')
line_2, = plt.plot(df3['HW'], color='red')
plt.legend([line_1, line_2], ['Initial Series', 'Smooth Series'])
plt.show()

df4 = df_for_last.copy()
mul_mul = ExponentialSmoothing(df4, seasonal_periods=12, trend='mul', seasonal='mul', use_boxcox=True, initialization_method='estimated',).fit()
df4['HW'] = list(mul_mul.fittedvalues)
plt.figure(figsize=(16,8))
plt.xlabel('Year')
plt.ylabel('Value')
plt.title('Multiplicative Trend & Multiplicative Seasonality')
line_1, = plt.plot(df4['Value'], color='blue')
line_2, = plt.plot(df4['HW'], color='red')
plt.legend([line_1, line_2], ['Initial Series', 'Smooth Series'])
plt.show()


fig, axs = plt.subplots(2, 2, figsize=(25,20))

axs[0, 0].plot(add_add.fittedvalues,color='red')
axs[0, 0].plot(df_for_last,color='blue')
axs[0, 0].set_title('Additive Trend $ Additive Seasonality')
axs[0, 0].set_xlabel('Year')
axs[0, 0].set_ylabel('Monthly anti-diabetic drug sales in Australia')

axs[0, 1].plot(add_mul.fittedvalues,color='red')
axs[0, 1].plot(df_for_last,color='blue')
axs[0, 1].set_title('Additive Trend $ Multiplicative Seasonality')
axs[0, 1].set_xlabel('Year')
axs[0, 1].set_ylabel('Monthly anti-diabetic drug sales in Australia')

axs[1, 0].plot(mul_add.fittedvalues,color='red')
axs[1, 0].plot(df_for_last,color='blue')
axs[1, 0].set_title('Multiplicative Trend $ Additive Seasonality')
axs[1, 0].set_xlabel('Year')
axs[1, 0].set_ylabel('Monthly anti-diabetic drug sales in Australia')

axs[1, 1].plot(mul_mul.fittedvalues,color='red')
axs[1, 1].plot(df_for_last,color='blue')
axs[1, 1].set_title('Multiplicative Trend $ Multiplicative Seasonality')
axs[1, 1].set_xlabel('Year')
axs[1, 1].set_ylabel('Monthly anti-diabetic drug sales in Australia')

plt.show()

