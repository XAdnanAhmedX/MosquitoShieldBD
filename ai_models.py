
from typing import Tuple, Dict, Any, Union, List
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

from data_generator import MEDICINES

def preprocess_data(df: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler]:

    features = ["Temperature", "Rainfall", "Humidity"]
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df[features])
    return scaled_array, scaler

def train_outbreak_model(df: pd.DataFrame, scaler: StandardScaler) -> LogisticRegression:
    features = ["Temperature", "Rainfall", "Humidity"]
    X = scaler.transform(df[features])
    y = (df["Actual_Cases"] > 50).astype(int)

    if len(np.unique(y)) < 2:
        y[0] = 1

    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X, y)
    return model

def train_symptom_model(symptom_df: pd.DataFrame) -> LogisticRegression:
    features = ["Fever", "JointPain", "Rash", "Headache", "EyePain"]
    X = symptom_df[features]
    y = symptom_df["Has_Disease"]

    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X, y)
    return model
