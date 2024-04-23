import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

data = pd.read_excel(r'C:\Users\Josemaria Escriva\Documents\beerec.xlsx')
data = data.dropna(axis=1, how='all')
data = data.dropna()

def load_data():
    dt = print(data.head())

    return dt

def desc():
    description = print('Data description:', data.describe())

    return description

def corre_fact():
    correlate = print('Correlation coefficients:', data.corr()['Optimal Harvest Time (0/1)'] )

    return correlate

def visualize():
    #create pair plot

    sns.pairplot(data)
    plt.show()

    # Create a heatmap of the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.show()

def predict():
   
    x = data.drop(columns=['Week of the Year','Avg. Daily Temp (°C)', 'Total Rainfall (mm)', 'Hive Weight Change (kg)', '% Capped Honey Frames', 'Norm. Bee Population'])
    Y = data['Optimal Harvest Time (0/1)']

    X_train, X_test, Y_train, Y_test = train_test_split(x,Y, test_size=0.35, random_state=42)
    model = LinearRegression()
    model.fit(X_train, Y_train)
    prediction = model.predict(X_test)
    MSE = print(mean_squared_error(Y_test, prediction))
    r2 = print(r2_score(Y_test, prediction))

    #test = print("Shape of x:", x.shape), print("Shape of Y:", Y.shape)
    
    return r2, MSE


predict()
load_data()
desc()
corre_fact()
visualize()  

