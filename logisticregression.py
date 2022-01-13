import pandas as pd
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

iterations = 25

hearts = pd.read_csv('hvectors.csv')
triangles = pd.read_csv('dvectors.csv')
circles = pd.read_csv('kvectors.csv')
squares = pd.read_csv('qvectors.csv')

hearts.drop(hearts.columns[0], axis=1, inplace=True)
triangles.drop(triangles.columns[0], axis=1, inplace=True)
circles.drop(circles.columns[0], axis=1, inplace=True)
squares.drop(squares.columns[0], axis=1, inplace=True)

hearts['2500'] = 0
triangles['2500'] = 1
circles['2500'] = 2
squares['2500'] = 3
