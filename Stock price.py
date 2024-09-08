import math
import tensorflow as tf
import numpy as np
import pandas as pd
import yfinance as yf
import pandas_datareader.data as web
from pandas_datareader import data as pdr
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

###Get the stock quote
df=yf.download("AAPL",start="2012-01-01",end="2023-12-17")
print(df)

#Get the number of rows and columns in the dataset
plt.figure(figsize=(16,8))
plt.title("Closed price history")
plt.plot(df["Close"])
plt.xlabel("Date",fontsize=18)
plt.ylabel("Close Price USD($)",fontsize=18)
plt.show()

##Create a new dataframe with only Close column
data=df.filter(["Close"])
##Convert the data frame to a numpy array
dataset=data.values
#get the number of rows to train the mode
training_data_len=math.ceil(len(dataset)*0.8)

print(training_data_len)

##Scale the data
scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(dataset)
print(scaled_data)

###Create training data set
###Create scaled training data set
train_data=scaled_data[0:training_data_len, :]
##split the data into x_train and y_train datasets
x_train=[]
y_train=[]
for i in range(60,len(train_data)):
    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])
    if i<=61:
        print(x_train)
        print(y_train)
        print("")

##Convert the x_train and y_train dataset to numpy arrays
x_train,y_train=np.array(x_train),np.array(y_train)

###Reshape the data
x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
print(x_train.shape)

##Build the LSTM model
model=Sequential()
model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

##Compile the model
model.compile(optimizer="adam",loss="mean_squared_error")

#train the model
model.fit(x_train,y_train,batch_size=1,epochs=1)

##Create the testing dataset
#Create new array containing scaled values from index 1543 to 2003
test_data=scaled_data[training_data_len-60: , :]
##Create the datasets x_test and y_test
x_test=[]
y_test=dataset[training_data_len-60:, :]
for i in range(60,len(test_data)):
    x_test.append(test_data[i-60:i,0])

#Convert the data
x_test=np.array(x_test)

#Reshape the data
x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

##Get the models predicted price values
predictions=model.predict(x_test)
predictions=scaler.inverse_transform(predictions)

predictions=np.append(predictions,np.zeros(60))


##Get the root mean square error(RMSE)
rmse=np.sqrt(np.mean(((predictions- y_test)**2)))
print(rmse)

##Plot the data

train=data[:training_data_len]
valid = data[training_data_len:]
predictions = predictions[:len(valid)]

# Assign predictions to the "Predictions" column
valid["Predictions"] = predictions

# Visualize data
plt.figure(figsize=(16, 8))
plt.title("Model")
plt.xlabel("Date", fontsize=18)
plt.ylabel("Close Price USD($)", fontsize=18)
plt.plot(train["Close"])
plt.plot(valid[["Close", "Predictions"]])
plt.legend(["Train", "Valid", "Predictions"], loc="lower right")
plt.show()

print(valid)

#Get the quote
apple_quote=yf.download("AAPL",start="2012-01-01",end="2024-6-1")
#Create a new dataframe
new_df=apple_quote.filter(["Close"])
##Get the last 60 day closing price values and convert the data frame to an array
last_60_days=new_df.tail(60)
#Scale the data to be values between 0 and 1
last_60_days_scaled=scaler.transform(last_60_days)
#Create an empty list
X_test=[]
##Append past 60 days to X_test list
X_test.append(last_60_days_scaled)
#Convert the X_test dataset to a numpy array
X_test=np.array(X_test)
#Reshape the data
X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
##Get the predicted scaled price
pred_price=model.predict(X_test)
##Undo the scaling
pred_price=scaler.inverse_transform(pred_price)
print(pred_price)

apple_quote2=yf.download("AAPL",start="2024-6-2",end="2024-6-2")
print(apple_quote2["Close"])
