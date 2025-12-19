
import datetime as dt
import time
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt

from plotly import graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import load_model  # use TF Keras consistently

# ---------------------------------
# Streamlit page config
# ---------------------------------
st.set_page_config(
    page_title="Stock Price Prediction System",
    page_icon="📈",
)

start = dt.datetime(2015, 1, 1)
end = dt.datetime.now()

st.title("📈 STOCK PRICE PREDICTION SYSTEM")

# ---------------------------------
# Ticker input
# ---------------------------------
ticker_input = st.text_input("Enter Stock ticker:", "^NSEI")
stockticker = yf.Ticker(ticker_input)

# ---------------------------------
# Data loading (uses passed ticker)
# ---------------------------------
@st.cache_data(show_spinner=True)
def Load_data(ticker_obj):
    df = ticker_obj.history(period="5y", auto_adjust=True)
    df = df.reset_index()
    # Basic health checks
    if df.empty:
        raise ValueError("No historical data returned. Try another symbol or period.")
    if "Close" not in df.columns:
        raise ValueError("No 'Close' column in returned data.")
    # Coerce Close to numeric and drop invalids
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Close"])
    if df.empty:
        raise ValueError("No valid closing prices (after cleaning).")
    return df

# ---------------------------------
# Fetch data + show sample
# ---------------------------------
st.subheader("Historical Data of past five years")
try:
    df = Load_data(stockticker)
except Exception as e:
    st.error(f"Failed to load data for {ticker_input}: {e}")
    st.stop()

st.write(df.tail())
df1 = df["Close"].astype(float).dropna()  # raw close series

# ---------------------------------
# Visualization: raw close
# ---------------------------------
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df1, name="Close"))
    fig.layout.update(title="Closing data chart", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)

plot_raw_data()

# ---------------------------------
# Split and scale (FIT ON TRAIN ONLY)
# ---------------------------------
vals = df1.values.reshape(-1, 1)
training_size = int(len(vals) * 0.65)
test_size = len(vals) - training_size

if training_size < 2 or test_size < 2:
    st.error("Not enough data to create train/test splits. Increase period.")
    st.stop()

train_data = vals[:training_size]
test_data = vals[training_size:]

scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)

# ---------------------------------
# Sequence builder
# ---------------------------------
@st.cache_data(show_spinner=False)
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    if len(dataset) <= time_step + 1:
        return np.array([]), np.array([])
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

# Time step
time_step = 100
X_train, y_train = create_dataset(train_scaled, time_step)
X_test, ytest = create_dataset(test_scaled, time_step)

if X_train.size == 0 or X_test.size == 0:
    st.error("Dataset too small for the selected time_step. Reduce time_step or increase period.")
    st.stop()

# Reshape to [samples, time_steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# ---------------------------------
# Load model (resource cached)
# ---------------------------------
@st.cache_resource(show_spinner=True)
def MY_Model(path="keras_model(1).h5"):
    try:
        model = load_model(path)
        return model
    except Exception as e:
        st.error(f"Failed to load model '{path}': {e}")
        return None

model = MY_Model()
if model is None:
    st.stop()

# ---------------------------------
# Predictions
# ---------------------------------
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse-transform predictions and labels
train_pred_inv = scaler.inverse_transform(train_predict)
test_pred_inv = scaler.inverse_transform(test_predict)

y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test_inv = scaler.inverse_transform(ytest.reshape(-1, 1))

# Sanity checks
if y_test_inv.shape[0] != test_pred_inv.shape[0]:
    st.error(
        f"Length mismatch: y_test_inv={y_test_inv.shape[0]} vs test_pred_inv={test_pred_inv.shape[0]}. "
        "Check time_step/dataset slicing."
    )
    st.stop()

# ---------------------------------
# Metrics (RMSE, MAE, MAPE, Accuracy)
# ---------------------------------
def calculate_mape(y_true, y_pred, eps=1e-8):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    mask = np.abs(y_true) > eps  # avoid divide-by-zero
    if np.sum(mask) == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# Flatten for metrics
y_true = y_test_inv.flatten()
y_hat = test_pred_inv.flatten()

rmse = float(np.sqrt(mean_squared_error(y_true, y_hat)))
mae = float(mean_absolute_error(y_true, y_hat))
mape = float(calculate_mape(y_true, y_hat))
accuracy_pct = 100 - mape  # interpretive accuracy

st.subheader("🔎 Test Set Performance")
col1, col2, col3 = st.columns(3)
col1.metric("RMSE (₹)", f"{rmse:,.2f}")
col2.metric("MAE (₹)", f"{mae:,.2f}")
col3.metric("Accuracy", f"{accuracy_pct:.2f}%")

def directional_accuracy(y_true_arr, y_pred_arr):
    if len(y_true_arr) < 2:
        return np.nan
    true_diff = np.diff(y_true_arr)
    pred_diff = np.diff(y_pred_arr)
    hits    hits = np.sum(np.sign(true_diff) == np.sign(pred_diff))
    return float(hits / len(true_diff) * 100)

dir_acc = directional_accuracy(y_true, y_hat)
st.caption(f"Directional Accuracy (up/down): {dir_acc:.1f}%")

# ---------------------------------
# Progress bar (optional)
# ---------------------------------
my_bar = st.progress(0)
for percent_complete in range(100):
    time.sleep(0.01)
    my_bar.progress(percent_complete + 1)
with st.spinner("Wait for a moment..."):
    time.sleep(0.1)

# ---------------------------------
# Plot: Actual vs Train/Test Predictions
# Build aligned series for plotting
# ---------------------------------
full_scaled = scaler.transform(vals)  # use train-fitted scaler on entire series
look_back = time_step

trainPredictPlot = np.empty_like(full_scaled)
trainPredictPlot[:] = np.nan
trainPredictPlot[look_back:len(train_pred_inv) + look_back, 0] = train_pred_inv[:, 0]

testPredictPlot = np.empty_like(full_scaled)
testPredictPlot[:] = np.nan
test_start = len(train_pred_inv) + (look_back * 2) + 1
test_end = test_start + len(test_pred_inv)
if test_start < len(full_scaled) and test_end <= len(full_scaled):
    testPredictPlot[test_start:test_end, 0] = test_pred_inv[:, 0]

st.subheader("Closing Price Graph with Training and Testing Data")
fig2 = plt.figure(figsize=(12, 6))
plt.plot(scaler.inverse_transform(full_scaled), label="Actual", color="blue")
plt.plot(trainPredictPlot, color="orange", label="Train Pred")
plt.plot(testPredictPlot, color="green", label="Test Pred")
plt.legend()
st.pyplot(fig2)

# ---------------------------------
# Forecast next 30 days using last window from test
# ---------------------------------
substract = len(test_scaled) - time_step
if substract < 0:
    st.warning("Not enough test data to build the last window for forecasting.")
    st.stop()

x_input = test_scaled[substract:].reshape(1, -1)
temp_input = list(x_input[0])

lst_output = []
n_steps = time_step
i = 0
while (i < 30):
    if (len(temp_input) > n_steps):
        x_input = np.array(temp_input[-n_steps:])  # always take last n_steps
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.append(yhat[0, 0])
        lst_output.append(yhat[0, 0])
        i += 1
    else:
        x_input = np.array(temp_input).reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.append(yhat[0, 0])
        lst_output.append(yhat[0, 0])
        i += 1

day_new = np.arange(1, time_step + 1)
day_pred = np.arange(time_step + 1, time_step + 31)

cnt = len(full_scaled) - time_step

st.subheader("Final Graph of prediction based on previous 100 days")
fig3 = plt.figure(figsize=(12, 6))
plt.plot(day_new, scaler.inverse_transform(full_scaled[cnt:cnt + time_step]).flatten(), label="Last 100 days")
plt.plot(day_pred, scaler.inverse_transform(np.array(lst_output).reshape(-1, 1)).flatten(), label="Forecast (30d)")
plt.legend()
st.pyplot(fig3)

# ---------------------------------
# Final combined series (Actual + Forecast)
# ---------------------------------
df3_scaled = np.vstack([full_scaled, np.array(lst_output).reshape(-1, 1)])
df3_inv = scaler.inverse_transform(df3_scaled).flatten()

st.subheader("Final prediction")
fig4 = plt.figure(figsize=(12, 6))
plt.plot(df3_inv, color="purple")
plt.title("Actual + Forecasted Series")
st.pyplot(fig4)

# ---------------------------------
# Footer
# ---------------------------------
st.markdown(
    "<footer><p>This app is for educational purposes only—use at your own risk.</p></footer>",
    unsafe_allow_html=True,
