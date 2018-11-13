import pandas
import numpy as np
import sklearn.linear_model as lm
from sklearn.model_selection import KFold
from sklearn import preprocessing as pre
import random
import matplotlib.pyplot as plt

housing = pandas.read_csv('housing.csv')

display(housing) #displays in Jupyter notebook