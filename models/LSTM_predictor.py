import keras
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
import pandas as pd 
import datetime
import numpy as np 
from matplotlib import style
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import os

os.chdir('D:/Scholastic/Projects/COVID-19_Jobs_Paper/LSTM_Predictor')

def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

### Model 1
d1 = pd.read_excel('Data/Unemployment_Rate_India_1991-2019.xlsx')
df1 = d1.rename(columns = {"Year": "ds", "Unemployment Rate":"y"})
df1['ds'] = pd.to_datetime(df1['ds'], yearfirst=True)

# interpolate data to get it for months
new_df1 = df1.set_index('ds').resample('M').interpolate(method='linear', axis=0, limit=None, inplace=False,\
         limit_direction='forward', limit_area=None, downcast=None).reset_index()


train_data = new_df1.loc[:,'y'].values
train_data = train_data.reshape(-1,1)

time_steps = 6
X_train, y_train = create_dataset(train_data, time_steps)

# reshape it [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], 6, 1))

print(X_train.shape)

# Build the model 
model1 = keras.Sequential()
model1.add(LSTM(units = 100, return_sequences = True, input_shape = (X_train.shape[1], 1)))
model1.add(Dropout(0.2))
model1.add(LSTM(units = 100))
model1.add(Dropout(0.2))
# Output layer
model1.add(Dense(units = 1))
# Compiling the model
model1.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the model to the Training set
history = model1.fit(X_train, y_train, epochs = 25, batch_size = 10, validation_split=0.1)

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='best')
plt.show()

preds_1 = model1.predict(X_train)

plt.plot(y_train, color = 'black', label = 'Actual')
plt.plot(preds_1, color = 'red', label = 'Predicted')
plt.xlabel('Time')
plt.ylabel('Unemployment Rate')
plt.legend()
plt.savefig('Model_1.jpg')



### Model 2
new_df2 = pd.read_csv('Data/Monthly_Unemployment_1991-2020.csv')
new_df2['ds'] = pd.to_datetime(new_df2['ds'], yearfirst=True)

train_data = new_df2.loc[:,'y'].values
train_data = train_data.reshape(-1,1)

time_steps = 6
X_train, y_train = create_dataset(train_data, time_steps)

# reshape it [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], 6, 1))

print(X_train.shape)

# Build the model 
model2 = keras.Sequential()
model2.add(LSTM(units = 100, return_sequences = True, input_shape = (X_train.shape[1], 1)))
model2.add(Dropout(0.2))
model2.add(LSTM(units = 100))
model2.add(Dropout(0.2))
# Output layer
model2.add(Dense(units = 1))
# Compiling the model
model2.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the model to the Training set
history = model2.fit(X_train, y_train, epochs = 25, batch_size = 10, validation_split=0.1)

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='best')
plt.show()

preds_2 = model2.predict(X_train)

plt.plot(y_train, color = 'black', label = 'Actual')
plt.plot(preds_2, color = 'red', label = 'Predicted')
plt.xlabel('Time')
plt.ylabel('Unemployment Rate')
plt.legend()
plt.savefig('Model_2.jpg')

