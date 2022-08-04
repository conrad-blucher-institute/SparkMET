import os
import os.path as path
import time

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
sns.set(font_scale=1.5)
from pprint import pprint
from IPython.display import HTML # to show the animation in Jupyter
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
#import geopandas as gpd
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import sklearn 
from sklearn.linear_model import LinearRegression
RANDOM_SEED = 6695
LAMBDA_TOLERANCE = 1e-10

import shap
shap.initjs()
