# Time-Series Forecasting on MSFT Stock Price
## Introduction 
This projects develops a deep learning moldel (LSTM/RNN) to forecast the Microsoft Corporation Common Stock (MSFT) price. Some pre-processing and exploratory data analysis was conducted first to explore the data. Four different models (SimpleRNN, 1-layer LSTM and 2-layer SLTM with different window sizes) were trained and compared for better performance. 

## Data 
The dataset used in this project is downloaded from https://www.nasdaq.com/market-activity/stocks/msft/historical?page=1&rows_per_page=10&timeline=y5. It contains MSFT stock prices from 26/05/2020 to 22/05/2022.

## Data Pre-processing and Exploratory Data Analysis
Pre-processing and exploratory data analysis were first conducted to explore the dataset and practice working with time-indexed data. 

## Model Construction
### Prepare the dataset. 
* Split the data into training/validation/test. The validation dataset will be used to evaluate different models. The final selected model is further trained on combined training and validaiton dataset. 
* Scale the datasets
* Transform the datasets using sliding windows
### Build and train RNN models. 
For a deep RNN architecture, we need to choose the depth (i.e, the number of layers the RNN has — e.g. stacking multiple LSTM or GRU layers on top of each other) and the neurons per layer (i.e., the number of units each RNN layer has — controls the layer’s capacity to learn patterns in sequences). 
Usually, a depth of 1–3 layers is sufficient for many problems. 
Typical number of units per layer are 32, 50, 64, 128, depending on data and compute. Too deep or too wide can cause overfitting or slow training — so we tune these by experimenting + validation.

Four different architectures (SimpleRNN, 1-layer LSTM and 2-layer SLTM with different window sizes) were experimented in this project. The 1-layer LSTM model with a window size of 60 performed best.
### Retrain the selected model.
 Combine the training dataset and validation dataset as the new full training dataset and go through the dataset preparation for LSTM models again before retraining the model.  
### Evaluate the model performance on test dataset
### Make predictions for the future
Future stock price prediction were made using prior 60 days stock price. Here each predicted value is fed back into the input for the next prediction. If a prediction is slightly off, that error propagates and often amplifies over subsequent steps. This causes the forecast to “drift” away from reality as we predict farther into the future. No mechanism here to adapt or retrain on new data as predictions move forward. 


## Summary results 

* A simple 1-layer LSTM architecture forcasts the *close price*  accurately with its prior 60 days *close price* as input, achieving a RMSE of 6.85 on the test data set. It has a RMSE of 6.85 on the test data set. This may or may not be acceptable depending on specific problems. Using log returns ($ \log \frac{P_t}{P_{t-1}} $) or price difference ($P_t-P_{t-1}$) are alternative methods for the stock price forcast. 

* We can extend this project for real-time streaming data by automating new data ingestion and retraining the model daily for the forcast of the next day's stock price.
