import cv2
import os
import os.path
import numpy as np
import math
import pandas as pd
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
from pprint import pprint
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
import pickle

np.random.seed(0)

TRAIN_DIR = 'Assets/Final/Train_base'
SUPER_TRAIN_DIR = 'Assets/Final/Train_super'
TEST_DIR = 'Assets/Final/Test'
CLASSES = os.listdir(TRAIN_DIR)

NUM_CLUSTERS = 40