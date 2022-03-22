import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

auto = pd.read_csv(r'auto-mpg.csv')
#auto = pd.read_csv(r'auto-mpg.csv', header=None, names=['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'car name'])
print(auto.shape)
print(auto.head())