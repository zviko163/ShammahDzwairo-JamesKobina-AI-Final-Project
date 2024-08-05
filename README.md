# ShammahDzwairo-JamesKobina-AI-Final-Project

**PROJECT DESCRIPTION**
Our project is a price forecasting machine learning model. it predicts the prices of a particular good/commodity based on the five previous unit prices that a user will enter. It predicts the future prices to up to 30 days and shows a visual representation of how the prices will be changing along the way. 

We trained three models, one LSTM whose hyperparameters where determined by the use of a grid search, a Birectional LSTM, and a Convolutional Neural Network model with one convolutional layer. We compared the three based on the loss, mean absolute error, and mean-squared error all averaged. The best performing model seemed to be the Bidirectional LSTM whic is the one we deployed using streamlit in python.

**HOW TO RUN THE APP**
Extract the folder and run the command: streamlit run app.py.
You will be directed to a webpage that displays our price prediction model.
