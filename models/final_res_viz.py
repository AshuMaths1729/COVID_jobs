import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_excel("data/Final_results.xlsx")
plt.plot(data['Dates'], data['ARIMA'])
plt.plot(data['Dates'], data['SARIMAX'])
plt.plot(data['Dates'], data['FARIMA'])
plt.plot(data['Dates'], data['Prophet'])
plt.plot(data['Dates'], data['LSTM'])
plt.plot(data['Dates'], data['SVR'])
plt.plot(data['Dates'], data['Model_v1'])
plt.plot(data['Dates'], data['Original'], color='-k')

plt.xticks(rotation=75)
plt.legend(['ARIMA', 'SARIMAX', 'FARIMA', 'Prophet', 'LSTM','SVR', 'Model_v1', 'Original'])
plt.savefig("Model_results.pdf")