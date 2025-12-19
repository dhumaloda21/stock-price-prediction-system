import datetime as dt
import time
import tensorflow
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.layers import LSTM


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader as data
import streamlit as st
from keras.models import load_model
from plotly import graph_objs as go
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.metrics import mean_squared_error
import yfinance as yf

st.set_page_config(
        page_title="Stock Price Prediction System",
        page_icon= '📈',
        
    )



start = dt.datetime(2015,1,1)
end = dt.datetime.now()
st.title(" STOCK PRICE PREDICTION SYSTEM ")
stockticker = yf.Ticker(st.text_input('Enter Stock ticker: ','^NSEI'))
#user_input = st.text_input('Enter Stock ticker: ',stockticker) 



def Load_data(ticker):
    df =stockticker.history(period='5y')
    df.reset_index(inplace=True)
    return df

#st.subheader('Historical Data of past five years  ')

##reseting index and choosing a column
st.subheader('Historical Data of past five years  ')
df = Load_data(stockticker)
st.write(df.tail())
df1=df['Close']


# Visiualization
def plot_raw_data():
    fig= go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'],y=df1,name='Stock_CLosed'))
    fig.layout.update(title='Closing data chart',xaxis_rangeslider_visible=True)

    #fig.update_layout(yaxis_range=[3000,30000])
    st.plotly_chart(fig)

plot_raw_data()

##
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))

##splitting dataset into train and test split
training_size=int(len(df1)*0.65)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]


# convert an array of values into a dataset matrix
@st.cache
def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return np.array(dataX), np.array(dataY)


##Setting a time step 
time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)


## Reshaping the training and testing data in 3 dimention 
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1],1 )
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],1 )



##Load model 
@st.cache(allow_output_mutation=True)
def MY_Model():
    model = load_model('keras_model(1).h5')
    return model


model = MY_Model()
### Lets Do the prediction and check performance metrics
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)

##Transformback to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)

##ADDED #####################################################################################################
# You should also inverse-transform y_train and y_test
y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
##y_test


from sklearn.metrics import mean_squared_error, mean_absolute_error

# ---- Helper: MAPE ----
def calculate_mape(y_true, y_pred):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    # Avoid division by zero
    mask = y_true != 0
    if np.sum(mask) == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# Flatten for metrics
y_true = y_test_inv.flatten()
y_hat  = test_predict.flatten()

# ---- Compute Metrics ----
rmse = float(np.sqrt(mean_squared_error(y_true, y_hat)))
mae  = float(mean_absolute_error(y_true, y_hat))
mape = float(calculate_mape(y_true, y_hat))
accuracy_pct = 100 - mape  # interpretive accuracy

# ---- Show in Streamlit ----
st.subheader("🔎 Test Set Performance")
col1, col2, col3 = st.columns(3)
col1.metric("RMSE (₹)", f"{rmse:,.2f}")
col2.metric("MAE (₹)",  f"{mae:,.2f}")
col3.metric("Accuracy", f"{accuracy_pct:.2f}%")

# (Optional) Directional accuracy (up/down correctness)
def directional_accuracy(y_true_arr, y_pred_arr):
    if len(y_true_arr) < 2: 
        return np.nan
    true_diff = np.diff(y_true_arr)
    pred_diff = np.diff(y_pred_arr)
    hits = np.sum(np.sign(true_diff) == np.sign(pred_diff))
    return float(hits / len(true_diff) * 100)

dir_acc = directional_accuracy(y_true, y_hat)
st.caption(f"Directional Accuracy (up/down)")
##################################################################################################


my_bar = st.progress(0)
for percent_complete in range(100):
    time.sleep(0.1)
    my_bar.progress(percent_complete + 1)
with st.spinner('Wait for sec......'):
    time.sleep(0.1)

#st.text("Mean Squared Error of Training data")
#st.write(math.sqrt(mean_squared_error(y_train,train_predict)))

#st.text("Mean Squared Error of Testing data")
#st.write(math.sqrt(mean_squared_error(ytest,test_predict)))



### Plotting 
# shift train predictions for plotting
look_back=100
trainPredictPlot = np.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = np.empty_like(df1)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict


# plot baseline and predictions
st.subheader('Closing Price Graph with Trainig and Testing Data')
fig2 = plt.figure(figsize=(12,6))
plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot,color='orange')
plt.plot(testPredictPlot,color='green')
st.pyplot(fig2)


substract=len(test_data)-100

x_input=test_data[substract:].reshape(1,-1)

temp_input=list(x_input)
temp_input=temp_input[0].tolist()


##prediction of next 30 days
#st.subheader("Predictions of next 30 days")
#input_numb= st.text_input('Enter the day:','1')
from numpy import array

lst_output=[]
n_steps=100
i=0
while(i<30):
    
    if(len(temp_input)>100):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        #st.write("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        #st.write("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        #st.write(yhat[0])
        temp_input.extend(yhat[0].tolist())
        #st.write(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
    
day_new=np.arange(1,101)
day_pred=np.arange(101,131)

cnt=len(df1)-100

st.subheader('Final Graph of prediction based on previous 100 days')
fig3 = plt.figure(figsize=(12,6))
plt.plot(day_new,scaler.inverse_transform(df1[cnt:]))
plt.plot(day_pred,scaler.inverse_transform(lst_output))
st.pyplot(fig3)


df3=df1.tolist()
df3.extend(lst_output)
plt.plot(df3[400:])

st.subheader('Final prediction')
fig4 = plt.figure(figsize=(12,6))
df3=scaler.inverse_transform(df3).tolist()
plt.plot(df3)
st.pyplot(fig4)

st.markdown('<footer><p>This app is for educational purpose follow on your risk</p> </footer>', unsafe_allow_html=True)
